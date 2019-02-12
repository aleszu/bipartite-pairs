
from scipy import sparse
import numpy as np
from abc import ABCMeta, abstractmethod  # enables abstract base classes

# use: my_model.akaike(adj_matrix)

class bipartiteGraphModel:
    __metaclass__ = ABCMeta

    # always use set_*_params() methods so we can keep track of num_params
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
        if (self.has_affil_params):
            num_params += len(self.affil_params)
        if (self.has_density_param):
            num_params += 1
        if (self.has_item_params):
            num_params += len(self.item_params)
        return num_params

    def print_params(self):
        if self.has_density_param:
            print "model intercept: " + str(round(self.density_param, 5))
        if self.has_item_params:
            print "item params: min " + str(round(np.min(self.item_params), 5)) + ", median " + \
                  str(round(np.median(self.item_params), 5)) + ", max " + str(
                round(np.max(self.item_params), 5))
        if self.has_affil_params:
            print "affil params: min " + str(round(np.min(self.affil_params), 5)) + ", median " + \
                  str(round(np.median(self.affil_params), 5)) + ", max " + str(
                round(np.max(self.affil_params), 5))

    # lower is better!
    def akaike(self, adj_matrix):
        loglik = self.loglikelihood(adj_matrix)
        aic = 2 * (self.get_num_params() - loglik)
        return aic

    # returns vector of log(P(edge_ij)) for item i
    @abstractmethod
    def loglik_edges_present(self, item_idx):
        pass

    @abstractmethod
    def loglik_edges_absent(self, item_idx):
        pass

    # higher is better!
    def loglikelihood(self, my_adj_mat):
        tot_score = 0
        # code borrowed from gen_all_pairs, to handle adj_matrices of different classes
        num_rows = my_adj_mat.shape[0]
        is_sparse = sparse.isspmatrix(my_adj_mat)
        is_numpy_matrix = (type(my_adj_mat) == np.matrix)
        for i in range(num_rows):
            if is_sparse:
                rowi = my_adj_mat.getrow(i).toarray()[0]  # toarray() gives 2-d matrix, [0] to flatten
            else:  # already an ndarray(), possibly a matrix()
                rowi = my_adj_mat[i,]
                if is_numpy_matrix:
                    rowi = rowi.A1

            # compute scores for edges that are present and absent, respectively
            score = rowi.dot(self.loglik_edges_present(i)) + (1 - rowi).dot(self.loglik_edges_absent(i))
            tot_score += score
        return tot_score



class bernoulliModel(bipartiteGraphModel):
    def __init__(self, pi_vector):
        self.set_affil_params(pi_vector)
        self.loglik_present = np.log(pi_vector)
        self.loglik_absent = np.log(1 - pi_vector)

    def loglik_edges_present(self, item_idx):
        return self.loglik_present

    def loglik_edges_absent(self, item_idx):
        return self.loglik_absent


class exponentialModel(bipartiteGraphModel):
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
        return np.log(e_expression / (1 + e_expression))


    # given an item ID, returns vector of log(1 - p_ij) = -log( 1 + exp(params) ) for all j
    def loglik_edges_absent(self, item_idx):
        e_expression = np.exp(self.density_param + self.item_params[item_idx] + self.affil_params)
        return np.log(1 / (1 + e_expression))

    # a dense matrix
    def edge_prob_matrix(self):
        e_expression = np.exp(self.item_params.reshape((-1, 1)) + self.affil_params.reshape((1, -1)) + self.density_param)
        return (e_expression) / (1 + e_expression)

    # if we don't want to instantiate the whole matrix at a time
    def edge_prob_row(self, item_idx):
        e_expression = np.exp(self.density_param + self.item_params[item_idx] + self.affil_params)
        return (e_expression / (1 + e_expression))

