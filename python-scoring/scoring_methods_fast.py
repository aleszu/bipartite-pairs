from timeit import default_timer as timer
import numpy as np
import pandas
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale
import scoring_methods



# Necessarily makes a dense matrix.
def simple_only_weighted_corr(pairs_generator, adj_matrix, pi_vector, print_timing=False):
    start = timer()
    transformed_mat = scoring_methods.wc_transform(adj_matrix, pi_vector)

    item1, item2, wc = [], [], []
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(transformed_mat):
        item1.append(row_idx1)
        item2.append(row_idx2)
        wc.append(pair_x.dot(pair_y))

    end = timer()
    print 'simple_only_weighted_corr: ' + str(end - start) + " secs" if print_timing else ''
    return pandas.DataFrame({'item1': item1, 'item2': item2, 'weighted_corr': wc})


# Leaves matrix sparse if it starts sparse
def simple_only_adamic_adar_scores(pairs_generator, adj_matrix, affil_counts, print_timing=False):
    start = timer()
    aa = []
    if sparse.isspmatrix(adj_matrix):
        transformed_mat = adj_matrix.multiply(1/np.sqrt(np.log(affil_counts))).tocsr()
    else:
        transformed_mat = adj_matrix / np.sqrt(np.log(affil_counts))
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(transformed_mat):
        aa.append(pair_x.dot(pair_y))
    end = timer()
    print 'simple_only_adamic_adar_scores: ' + str(end - start) + " secs" if print_timing else ''
    return aa

# Leaves matrix sparse if it starts sparse
# use_package=True has a slight speed edge over my implementation, when matrix is dense
def simple_only_cosine(pairs_generator, adj_matrix, weights=None, print_timing=False, use_package=False):
    start = timer()
    cos = []
    if weights is not None:
        if sparse.isspmatrix(adj_matrix):
            transformed_mat = adj_matrix.multiply(weights).tocsr()
        else:
            transformed_mat = adj_matrix * weights
    else:
        transformed_mat = adj_matrix

    if use_package:
        all_pairs_scores = cosine_similarity(transformed_mat)
        for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(transformed_mat):
            score = all_pairs_scores[row_idx1, row_idx2]
            cos.append(score if not np.isnan(score) else 0)

    else:  # also implement it myself
        # make each row have unit length
        if sparse.isspmatrix(transformed_mat):
            row_norms = 1/np.sqrt(transformed_mat.power(2).sum(axis=1).A1)
            # To multiply each row of a matrix by a different number, put the numbers into a diagonal matrix to the left
            transformed_mat = sparse.spdiags(row_norms, 0, len(row_norms), len(row_norms)) * transformed_mat
        else:
            row_norms = np.sqrt((transformed_mat * transformed_mat).sum(axis=1))
            transformed_mat = transformed_mat / row_norms.reshape(-1,1)

        for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(transformed_mat):
            dot_prod = pair_x.dot(pair_y)
            cos.append(dot_prod if not np.isnan(dot_prod) else 0)

    end = timer()
    print 'simple_only_cosine: ' + str(end - start) + " secs" if print_timing else ''
    return cos



# Leaves sparse matrix sparse
def simple_only_pearson(pairs_generator, adj_matrix, print_timing=False):
    start = timer()

    # scale rows to have std 1.
    # sklearn.preprocessing.scale will only take CSC sparse matrices and axis 0, so need to transpose back and
    # forth (cheap operation) to get my CSR with axis 1
    transformed_matrix = scale(adj_matrix.transpose().astype(float, copy=False),  # cast to float to avoid warning msg
                               axis=0, with_mean=False, with_std=True).transpose()

    row_means = transformed_matrix.mean(axis=1)
    if type(row_means) == np.matrix:
        row_means = row_means.A1

    n = adj_matrix.shape[1]
    scores = []
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(transformed_matrix):
        scores.append((pair_x - row_means[row_idx1]).dot(pair_y - row_means[row_idx2]) / float(n))

    end = timer()
    print 'simple_only_pearson: ' + str(end - start) + " secs" if print_timing else ''
    return scores


def simple_weighted_corr_sparse(pairs_generator, adj_matrix, pi_vector, print_timing=False):
    start = timer()

    terms_for_11 = (1 - pi_vector) / pi_vector
    #sqrt_terms_for_10 = np.full_like(pi_vector, fill_value=-1)
    terms_for_00 = pi_vector / (1 - pi_vector)

    wc = []
    n = float(adj_matrix.shape[1])
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(adj_matrix):
        sum_11 = (pair_x * pair_y).dot(terms_for_11)
        sum_00 = (np.logical_not(pair_x) * np.logical_not(pair_y)).dot(terms_for_00)
        sum_10 = -np.logical_xor(pair_x, pair_y).sum()
        wc.append((sum_11 + sum_10 + sum_00) / n)

    end = timer()
    print 'simple_weighted_corr_sparse: ' + str(end - start) + " secs" if print_timing else ''
    return wc



# Sparsity type rules (notes to self):
# -Each function says whether it's forced to create a dense matrix
# -A scipy.sparse made dense turns into a matrix(), while if I convert it to dense outside, it'll be (and stay) an ndarray

