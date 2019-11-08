from __future__ import print_function
from builtins import str, range, object
from scipy import sparse
import numpy as np
from abc import ABCMeta, abstractmethod  # enables abstract base classes
from future.utils import with_metaclass
from timeit import default_timer as timer

# todo: check if num_params is correct for akaike. Might need to be len(affil_params) - 1 and len(item_params) - 1.


class BipartiteGraphModel(with_metaclass(ABCMeta, object)):
    has_affil_params = False
    has_item_params = False
    has_density_param = False

    def set_affil_params(self, pi_vector):
        self.affil_params = pi_vector
        self.has_affil_params = True

    def set_item_params(self, item_params):
        self.item_params = item_params
        self.has_item_params = True

    def set_density_param(self, dens):
        self.density_param = dens
        self.has_density_param = True

    def get_num_params(self):
        num_params = 0
        if self.has_affil_params:
            num_params += len(self.affil_params)
        if self.has_density_param:
            num_params += 1
        if self.has_item_params:
            num_params += len(self.item_params)
        return num_params

    def print_params(self):
        if self.has_density_param:
            print("model intercept: " + str(round(self.density_param, 5)))
        if self.has_item_params:
            print("item params: min " + str(round(np.min(self.item_params), 5)) + ", median " + \
                  str(round(np.median(self.item_params), 5)) + ", max " + str(
                round(np.max(self.item_params), 5)))
        if self.has_affil_params:
            print("affil params: min " + str(round(np.min(self.affil_params), 5)) + ", median " + \
                  str(round(np.median(self.affil_params), 5)) + ", max " + str(
                round(np.max(self.affil_params), 5)))

    # returns vector of log(P(edge_ij)) for item i
    @abstractmethod
    def loglik_edges_present(self, item_idx):
        pass

    @abstractmethod
    def loglik_edges_absent(self, item_idx):
        pass

    # For efficiency, have a single function compute and return indiv item log likelihoods, total data
    # log likelihood (higher is better fit), and akaike (lower is better fit)
    def likelihoods(self, my_adj_mat, print_timing=False):
        start = timer()
        # code borrowed from gen_all_pairs, to handle adj_matrices of different classes
        is_sparse = sparse.isspmatrix(my_adj_mat)
        # is_numpy_matrix = (type(my_adj_mat) == np.matrix)
        # if (not is_sparse):     #and (not is_numpy_matrix):
        #     return self.likelihoods_dense(my_adj_mat, print_timing)

        num_rows = my_adj_mat.shape[0]
        item_LLs = np.zeros(num_rows)
        loglik = 0
        for i in range(num_rows):
            if is_sparse:
                rowi = my_adj_mat.getrow(i).toarray()[0]  # toarray() gives 2-d ndarray, [0] to flatten
            else:  # already an ndarray(), possibly a matrix()
                rowi = my_adj_mat[i,]
                # if is_numpy_matrix:
                #     rowi = rowi.A1

            # compute scores for edges that are present and absent, respectively
            # score = rowi.dot(self.loglik_edges_present(i)) + (1 - rowi).dot(self.loglik_edges_absent(i))    # v0, fast but spreads nans
            # score = self.loglik_edges_present(i)[rowi == 1].sum() + self.loglik_edges_absent(i)[rowi==0].sum() # v2, slower
            score = rowi.choose([self.loglik_edges_absent(i), self.loglik_edges_present(i)], mode="clip").sum()    # v4, still slow
            loglik += score
            item_LLs[i] = score

        aic = 2 * (self.get_num_params() - loglik)
        end = timer()
        if print_timing:
            print("computed likelihoods in " + str(end - start) + " seconds")
        return (loglik, aic, item_LLs)

    # inactive, since slower than the regular version
    def likelihoods_dense(self, my_adj_mat, print_timing=False):
        start = timer()
        # log(edge_probs) dot adj_mat  --> LL per item from edges present
        # log(1 - edge_probs) * (1 - adj_mat) --> LL per item from edges absent
        # I want each rowi.dot(rowi). Not matrix multiplication, which is what happens with matrix.dot(matrix).
        # So use rowsums of element-wise multiplication.
        LL_present = np.sum(np.log(self.edge_prob_matrix()) * my_adj_mat, axis=1)
        LL_absent = np.sum(np.log(1 - self.edge_prob_matrix()) * (1 - my_adj_mat), axis=1)
        item_LLs = LL_present + LL_absent
        loglik = np.sum(item_LLs)
        aic = 2 * (self.get_num_params() - loglik)

        end = timer()
        if print_timing:
            print("computed likelihoods (w/matrices) in " + str(end - start) + " seconds")

        return loglik, aic, item_LLs

    # higher is better!
    def loglikelihood(self, my_adj_mat):
        return self.likelihoods(my_adj_mat)[0]

    # lower is better!
    def akaike(self, adj_matrix):
        return self.likelihoods(adj_matrix)[1]


class bernoulliModel(BipartiteGraphModel):
    def __init__(self, pi_vector):
        self.set_affil_params(pi_vector)
        with np.errstate(divide='ignore'):  # don't warn for log(0) (= -inf)
            self.loglik_present = np.log(pi_vector)
            self.loglik_absent = np.log(1 - pi_vector)

    def loglik_edges_present(self, item_idx):
        return self.loglik_present

    def loglik_edges_absent(self, item_idx):
        return self.loglik_absent

    # returns 1 edge_prob_row, so caller must broadcast it to the desired number of rows
    def edge_prob_matrix(self):
        return self.affil_params


class ExponentialModel(BipartiteGraphModel):
    exp_model_converged = False
    # likelihood of edge (i,j):
    # log(p_ij / (1 - p_ij)) = density_param + item_params_i + affil_params_j
    #
    # so, p_ij =  exp( density_param + item_params_i + affil_params_j ) / 1 + exp(same params)
    #
    # likelihood of full graph (no additional normalization needed) = product (over all edges): p_ij
    #
    # Note that density_param is the same as logistic regression's intercept term

    def __init__(self, num_items, num_affils):
        # don't use set_* to initialize these, because they might stay blank
        self.item_params = np.zeros(num_items)
        self.affil_params = np.zeros(num_affils)
        self.density_param = 0

    # given an item ID, returns vector of log(p_ij) for all j
    def loglik_edges_present(self, item_idx):
        e_expression = np.exp(self.density_param + self.item_params[item_idx] + self.affil_params)
        with np.errstate(divide='ignore', invalid='ignore'):  # don't warn for log(0) (= -inf), nor for the inf/inf that we don't use
            return np.where(np.isinf(e_expression), 0, np.log(e_expression / (1 + e_expression)))

    # given an item ID, returns vector of log(1 - p_ij) = -log( 1 + exp(params) ) for all j
    def loglik_edges_absent(self, item_idx):
        e_expression = np.exp(self.density_param + self.item_params[item_idx] + self.affil_params)
        with np.errstate(divide='ignore'):  # don't warn for log(1/inf) = log(0) = -inf
            return np.log(1 / (1 + e_expression))

    # a dense matrix
    def edge_prob_matrix(self):
        e_expression = np.exp(self.item_params.reshape((-1, 1)) + self.affil_params.reshape((1, -1)) + self.density_param)
        with np.errstate(invalid='ignore'):
            return np.where(np.isinf(e_expression), 1, (e_expression) / (1 + e_expression))

    # if we don't want to instantiate the whole matrix at a time
    def edge_prob_row(self, item_idx):
        e_expression = np.exp(self.density_param + self.item_params[item_idx] + self.affil_params)
        with np.errstate(invalid='ignore'):
            return np.where(np.isinf(e_expression), 1, (e_expression / (1 + e_expression)))

