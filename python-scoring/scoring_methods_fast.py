from __future__ import print_function
from builtins import str
from timeit import default_timer as timer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.preprocessing import scale
import transforms_for_dot_prods

# Leaves matrix sparse if it starts sparse.
# Note that the package's call computes scores among all pairs of rows -- may not always be what we want.
# sklearn's cosine_similarity is really fast (uses C). For sparse matrices, it's super memory efficient, but for dense
# matrices, it's super memory inefficient. use_batches allows a better tradeoff for dense matrices.
# sklearn (use_package=True) is far better than my implementation, as data scales.
def simple_only_cosine(pairs_generator, adj_matrix, weights=None, print_timing=False, use_package=False, use_batches=False):
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
        if use_batches and not sparse.isspmatrix(adj_matrix):
            all_pairs_scores = cosine_similarity_n_space(transformed_mat, transformed_mat, verbose=print_timing)
        else:
            all_pairs_scores = cosine_similarity(transformed_mat)
        for (row_idx1, row_idx2, _, _, _, _) in pairs_generator(transformed_mat):
            score = all_pairs_scores[row_idx1, row_idx2]
            cos.append(score if not np.isnan(score) else 0)

    else:  # also implement it myself
        # make each row have unit length
        transformed_mat = transforms_for_dot_prods.cosine_transform(transformed_mat)

        for (_, _, _, _, pair_x, pair_y) in pairs_generator(transformed_mat):
            dot_prod = pair_x.dot(pair_y)
            cos.append(dot_prod if not np.isnan(dot_prod) else 0)

    end = timer()
    if print_timing:
        print('simple_only_cosine: ' + str(end - start) + " secs")
    return cos

# taken from https://stackoverflow.com/a/45202988/1014857
def cosine_similarity_n_space(m1, m2, batch_size=500, verbose=False):
    assert m1.shape[1] == m2.shape[1]
    if verbose:
        print("using batch size of " + str(batch_size))
    ret = np.ndarray((m1.shape[0], m2.shape[0]))
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
    return ret

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
    for (row_idx1, row_idx2, _, _, pair_x, pair_y) in pairs_generator(transformed_matrix):
        scores.append((pair_x - row_means[row_idx1]).dot(pair_y - row_means[row_idx2]) / float(n))

    end = timer()
    if print_timing:
        print('simple_only_pearson: ' + str(end - start) + " secs")
    return scores


# Sparsity type rules (notes to self):
# -Each function's comments say whether it's forced to create a dense matrix
# -A scipy.sparse made dense turns into a matrix(), while if I convert it to dense outside, it'll be (and stay) an ndarray

