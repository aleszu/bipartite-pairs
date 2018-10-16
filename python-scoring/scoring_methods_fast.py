from timeit import default_timer as timer
import numpy as np
import pandas
from scipy import sparse

# Turns out the scoring functions are 10x faster when the function can be written as: transform the adjacency matrix, then
# take dot products of the row pairs.
def compute_scores_fast(pairs_generator, adj_matrix, transf_func, print_timing=False, **named_args_to_func):
    start = timer()
    transformed_mat = transf_func(adj_matrix, **named_args_to_func)
    is_sparse = sparse.isspmatrix(transformed_mat)
    scores = []
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(transformed_mat):
        if is_sparse:
            scores.append(np.dot(pair_x, pair_y))
        else:
            scores.append(np.dot(pair_x, pair_y.reshape(max(pair_y.shape), 1))[0, 0])
    end = timer()
    print transf_func.__name__ + ": " + str(end - start) + " secs" if print_timing else ''
    return scores

# Standardize adj_matrix column-wise: for each coln, x --> (x - p_i) / sqrt(p_i(1-p_i))
# Also multiply denominator by sqrt(num_cols), so that the final dot product returns the mean, not just a sum
def wc_transform(adj_matrix, pi_vector):
    return (adj_matrix - pi_vector) * 1 / np.sqrt(pi_vector * (1 - pi_vector) * adj_matrix.shape[1])

def adamic_adar_transform(adj_matrix, pi_vector, num_docs):
    affil_counts = np.maximum(num_docs * pi_vector, 2)
    return adj_matrix / np.sqrt(np.log(affil_counts))

def newman_transform(adj_matrix, pi_vector, num_docs):
    affil_counts = np.maximum(num_docs * pi_vector, 2)
    return adj_matrix / np.sqrt(affil_counts.astype(float) - 1)

def shared_weight11_transform(adj_matrix, pi_vector):
    return adj_matrix.multiply(np.sqrt(np.log(1/pi_vector))).tocsr()

def simple_only_weighted_corr(pairs_generator, adj_matrix, pi_vector, print_timing=False):
    start = timer()
    # standardize adj_matrix column-wise: for each coln, x --> (x - p_i) / sqrt(p_i(1-p_i))
    transformed_mat = (adj_matrix - pi_vector) * 1 / np.sqrt(pi_vector * (1 - pi_vector) * adj_matrix.shape[1])

    item1, item2, wc = [], [], []
    num_cols = transformed_mat.shape[1]
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(transformed_mat):
        item1.append(row_idx1)
        item2.append(row_idx2)
        wc.append(pair_x.dot(pair_y.reshape(num_cols,1))[0, 0])

    end = timer()
    print 'simple_only_weighted_corr: ' + str(end - start) + " secs" if print_timing else ''
    return pandas.DataFrame({'item1': item1, 'item2': item2, 'weighted_corr': wc})


def simple_only_adamic_adar_scores(pairs_generator, adj_matrix, affil_counts, print_timing=False):
    start = timer()
    aa = []
    transformed_mat = adj_matrix / np.sqrt(np.log(affil_counts))
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(transformed_mat):
        aa.append(pair_x.dot(pair_y.reshape(pair_y.shape[1], 1)))
    end = timer()
    print 'simple_only_adamic_adar_scores: ' + str(end - start) + " secs" if print_timing else ''
    return aa
