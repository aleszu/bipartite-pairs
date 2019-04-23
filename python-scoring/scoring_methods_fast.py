from __future__ import print_function
from builtins import str
from timeit import default_timer as timer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.preprocessing import scale
import transforms_for_dot_prods
import scoring_methods

# Leaves matrix sparse if it starts sparse.
# Note that the package's call computes scores among all pairs of rows -- may not always be what we want.
# sklearn's cosine_similarity is really fast (uses C). For sparse matrices, it's super memory efficient, but for dense
# matrices, it's super memory inefficient. use_batches allows a better tradeoff for dense matrices.
# sklearn (use_package=True) is far better than my implementation, as data scales.
def simple_only_cosine(pairs_generator, adj_matrix, scores_out, weights=None, print_timing=False, use_package=False, use_batches=False):
    start = timer()
    if weights is not None:
        if sparse.isspmatrix(adj_matrix):
            transformed_mat = adj_matrix.multiply(weights).tocsr()
        else:
            transformed_mat = adj_matrix * weights
    else:
        transformed_mat = adj_matrix

    if use_package:
        if use_batches and not sparse.isspmatrix(adj_matrix):
            cosine_similarity_n_space(transformed_mat, transformed_mat, scores_out, verbose=print_timing)
        else:
            scores_out[:] = cosine_similarity(transformed_mat)[:]

    else:  # also implement it myself
        # make each row have unit length
        transformed_mat = transforms_for_dot_prods.cosine_transform(transformed_mat)

        for (i, j, _, _, pair_x, pair_y) in pairs_generator(transformed_mat):
            dot_prod = pair_x.dot(pair_y)
            scores_out[i,j] = (dot_prod if not np.isnan(dot_prod) else 0)

    end = timer()
    if print_timing:
        print('simple_only_cosine: ' + str(end - start) + " secs")

# taken from https://stackoverflow.com/a/45202988/1014857
def cosine_similarity_n_space(m1, m2, ret, batch_size=500, verbose=False):
    assert m1.shape[1] == m2.shape[1]
    if verbose:
        print("using batch size of " + str(batch_size))
    # ret = np.ndarray((m1.shape[0], m2.shape[0]))
    cnt = 1
    for row_i in range(0, int(m1.shape[0] / batch_size) + 1):
        start = row_i * batch_size
        end = min([(row_i + 1) * batch_size, m1.shape[0]])
        if end <= start:
            break   # edge cases
        if verbose and (cnt % 10 == 0):
            print("\tbatch " + str(cnt))
        rows = m1[start: end]
        sim = cosine_similarity(rows, m2) # rows is O(1) size
        ret[start: end] = sim
        cnt += 1
    # return ret

# Leaves sparse matrix sparse
def simple_only_pearson(pairs_generator, adj_matrix, scores_out, print_timing=False):
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
    for (row_idx1, row_idx2, _, _, pair_x, pair_y) in pairs_generator(transformed_matrix):
        scores_out[row_idx1,row_idx2] = ((pair_x - row_means[row_idx1]).dot(pair_y - row_means[row_idx2]) / float(n))

    end = timer()
    if print_timing:
        print('simple_only_pearson: ' + str(end - start) + " secs")



# jaccard = shared_size(i,j) / (rowi.sum() + rowj.sum() - shared_size(i,j))
def jaccard_from_sharedsize(pairs_generator, adj_matrix, scores_storage, scores_out, print_timing = False, back_compat = False):
    start = timer()

    if "shared_size" in scores_storage.underlying_dict:
        ss_scores = scores_storage.retrieve_array("shared_size")
    else:
        # Betting that computing the scores first is faster than doing without them
        ss_scores = scores_storage.create_and_store_unofficial("shared_size", dtype=int if back_compat else float)
        scoring_methods.compute_scores_with_transform(pairs_generator, adj_matrix,
                                      transforms_for_dot_prods.shared_size_transform, ss_scores,
                                      print_timing=print_timing, back_compat=back_compat)

    rowsums = np.asarray(adj_matrix.sum(axis=1)).squeeze()
    # create a square matrix with rowsums along each row
    # rowsums_mat = rowsums[:, np.newaxis] + np.zeros(adj_matrix.shape[0])

    if back_compat:
        scores_out[:] = (ss_scores.astype('float') / (rowsums[:, np.newaxis] + rowsums[np.newaxis, :] - ss_scores))[:]
    else:
        scores_out[:] = ((ss_scores * adj_matrix.shape[1]).round() /
                         (rowsums[:, np.newaxis] + rowsums[np.newaxis, :] - (ss_scores * adj_matrix.shape[1]).round()))[:]

    # fix NaNs, which occur where both rowsums are 0
    row_0s = np.flatnonzero(rowsums == 0)
    for i in row_0s:
        for j in row_0s:
            scores_out[i,j] = 0

    end = timer()
    if print_timing:
        print("jaccard_from_sharedsize: " + str(end - start) + " secs")


# hamming is equivalent to rowi.sum() + rowj.sum() - 2 * shared_size(i,j)
# If not back_compat, then every term gets divided by num affils, and we take (1 - x) to make it similarity, not distance.
def hamming_from_sharedsize(pairs_generator, adj_matrix, scores_storage, scores_out, print_timing = False, back_compat = False):
    start = timer()

    if "shared_size" in scores_storage.underlying_dict:
        ss_scores = scores_storage.retrieve_array("shared_size")
    else:
        # Betting that computing the scores first is faster than doing without them
        ss_scores = scores_storage.create_and_store_unofficial("shared_size", dtype=int if back_compat else float)
        scoring_methods.compute_scores_with_transform(pairs_generator, adj_matrix,
                                      transforms_for_dot_prods.shared_size_transform, ss_scores,
                                      print_timing=print_timing, back_compat=back_compat)

    rowsums = np.asarray(adj_matrix.sum(axis=1)).squeeze()
    # create a square matrix with rowsums along each row
    # rowsums_mat = rowsums[:, np.newaxis] + np.zeros(adj_matrix.shape[0])

    if back_compat:
        scores_out[:] = (rowsums[:, np.newaxis] + rowsums[np.newaxis, :] - 2 * ss_scores)[:]
    else:  # keeping things as integers until the last step, to preserve floating point precision
        scores_out[:] = ((adj_matrix.shape[1] -
                         (rowsums[:, np.newaxis] + rowsums[np.newaxis, :] - 2 * (ss_scores * adj_matrix.shape[1]).round())
                          ) / float(adj_matrix.shape[1]))[:]

    end = timer()
    if print_timing:
        print("hamming_from_sharedsize: " + str(end - start) + " secs")

# Sparsity type rules (notes to self):
# -Each function's comments say whether it's forced to create a dense matrix
# -A scipy.sparse made dense turns into a matrix(), while if I convert it to dense outside, it'll be (and stay) an ndarray
