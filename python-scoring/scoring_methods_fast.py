from timeit import default_timer as timer
import numpy as np
import pandas
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# Turns out the scoring functions can be ~10x faster when the function is written as: transform the adjacency matrix, then
# take dot products of the row pairs.
def compute_scores_fast(pairs_generator, adj_matrix, transf_func, print_timing=False, **named_args_to_func):
    start = timer()
    transformed_mat = transf_func(adj_matrix, **named_args_to_func)
    scores = []
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(transformed_mat):
        scores.append(pair_x.dot(pair_y))
    end = timer()
    print transf_func.__name__ + ": " + str(end - start) + " secs" if print_timing else ''
    return scores

# Standardize adj_matrix column-wise: for each coln, x --> (x - p_i) / sqrt(p_i(1-p_i))
# Also multiply denominator by sqrt(num_cols), so that the final dot product returns the mean, not just a sum.
# Necessarily makes a dense matrix.
def wc_transform(adj_matrix, pi_vector):
    return (adj_matrix - pi_vector) / np.sqrt(pi_vector * (1 - pi_vector) * adj_matrix.shape[1])

# Leaves matrix sparse if it starts sparse
def adamic_adar_transform(adj_matrix, pi_vector, num_docs):
    affil_counts = np.maximum(num_docs * pi_vector, 2)
    if sparse.isspmatrix(adj_matrix):
        return adj_matrix.multiply(1/np.sqrt(np.log(affil_counts))).tocsr()
    else:
        return adj_matrix / np.sqrt(np.log(affil_counts))

# Leaves matrix sparse if it starts sparse
def newman_transform(adj_matrix, pi_vector, num_docs):
    affil_counts = np.maximum(num_docs * pi_vector, 2)
    if sparse.isspmatrix(adj_matrix):
        return adj_matrix.multiply(1/np.sqrt(affil_counts.astype(float) - 1)).tocsr()
    else:
        return adj_matrix / np.sqrt(affil_counts.astype(float) - 1)

# Leaves matrix sparse if it starts sparse
def shared_size_transform(adj_matrix):
    return adj_matrix

# Leaves matrix sparse if it starts sparse
def shared_weight11_transform(adj_matrix, pi_vector):
    if sparse.isspmatrix(adj_matrix):
        # Keep the matrix sparse if it was before. (By default changes to coo() if I don't cast it tocsr().)
        return adj_matrix.multiply(np.sqrt(np.log(1 / pi_vector))).tocsr()  # .multiply() doesn't exist if adj_matrix is dense
    else:
        return adj_matrix * np.sqrt(np.log(1 / pi_vector))  # gives incorrect behavior if adj_matrix is sparse

# Necessarily makes a dense matrix.
def simple_only_weighted_corr(pairs_generator, adj_matrix, pi_vector, print_timing=False):
    start = timer()
    transformed_mat = wc_transform(adj_matrix, pi_vector)

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


# Sparsity type rules (notes to self):
# -Each function says whether it's forced to create a dense matrix
# -A scipy.sparse made dense turns into a matrix(), while if I convert it to dense outside, it'll be (and stay) an ndarray

