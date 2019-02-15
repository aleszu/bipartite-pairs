from scipy import sparse
import numpy as np
from sklearn.preprocessing import scale


# Standardize adj_matrix column-wise: for each coln, x --> (x - p_i) / sqrt(p_i(1-p_i))
# Also multiply denominator by sqrt(num_cols), so that the final dot product returns the mean, not just a sum.
# Necessarily makes a dense matrix.
def wc_transform(adj_matrix, pi_vector):
    return (adj_matrix - pi_vector) / np.sqrt(pi_vector * (1 - pi_vector) * adj_matrix.shape[1])


# See comments for wc_transform
def wc_exp_transform(adj_matrix, exp_model):
    edge_probs = exp_model.edge_prob_matrix()
    return (adj_matrix - edge_probs) / np.sqrt(edge_probs * (1 - edge_probs) * adj_matrix.shape[1])


# Leaves matrix sparse if it starts sparse
def newman_transform(adj_matrix, pi_vector, num_docs, make_dense=False):
    affil_counts = np.maximum(num_docs * pi_vector, 2)
    if sparse.isspmatrix(adj_matrix) and not make_dense:
        return adj_matrix.multiply(1 / np.sqrt(affil_counts.astype(float) - 1)).tocsr()
    else:
        return adj_matrix / np.sqrt(affil_counts.astype(float) - 1)


# Leaves matrix sparse if it starts sparse
def shared_size_transform(adj_matrix, back_compat=False, make_dense=False):
    if back_compat:
        return adj_matrix
    else:  # todo: test this version
        if sparse.isspmatrix(adj_matrix) and not make_dense:
            return adj_matrix.multiply(1 / np.sqrt(adj_matrix.shape[1]))
        else:
            return adj_matrix / np.sqrt(adj_matrix.shape[1])


# Leaves matrix sparse if it starts sparse
def shared_weight11_transform(adj_matrix, pi_vector, make_dense=False):
    if sparse.isspmatrix(adj_matrix) and not make_dense:
        # Keep the matrix sparse if it was before. (By default changes to coo() if I don't cast it tocsr().)
        return adj_matrix.multiply(
            np.sqrt(np.log(1 / pi_vector))).tocsr()  # .multiply() doesn't exist if adj_matrix is dense
    else:
        return adj_matrix * np.sqrt(np.log(1 / pi_vector))  # gives incorrect behavior if adj_matrix is sparse


# Leaves matrix sparse if it starts sparse
def adamic_adar_transform(adj_matrix, pi_vector, num_docs, make_dense=False):
    affil_counts = np.maximum(num_docs * pi_vector, 2)
    if sparse.isspmatrix(adj_matrix) and not make_dense:
        return adj_matrix.multiply(1 / np.sqrt(np.log(affil_counts))).tocsr()
    else:
        return adj_matrix / np.sqrt(np.log(affil_counts))


# Leaves matrix sparse if it starts sparse
def cosine_transform(adj_matrix, make_dense=False):
    if sparse.isspmatrix(adj_matrix) and make_dense:
        adj_matrix = adj_matrix.toarray()

    if sparse.isspmatrix(adj_matrix):
        row_norms = 1 / np.sqrt(adj_matrix.power(2).sum(axis=1).A1)
        row_norms[np.isinf(row_norms)] = 0
        # To multiply each row of a matrix by a different number, put the numbers into a diagonal matrix to the left
        transformed_mat = sparse.spdiags(row_norms, 0, len(row_norms), len(row_norms)) * adj_matrix
    else:
        row_norms = np.sqrt((adj_matrix * adj_matrix).sum(axis=1))
        row_norms[np.isinf(row_norms)] = 0
        transformed_mat = adj_matrix / row_norms.reshape(-1, 1)
    return transformed_mat


# Leaves matrix sparse if it starts sparse
def cosine_weights_transform(adj_matrix, weights, make_dense=False):
    if sparse.isspmatrix(adj_matrix):
        transformed_mat = adj_matrix.multiply(weights).tocsr()
    else:
        transformed_mat = adj_matrix * weights

    return cosine_transform(transformed_mat, make_dense)



# Resulting matrix has to be dense
def pearson_transform(adj_matrix):
    # scale rows to have std 1.
    # sklearn.preprocessing.scale will only take CSC sparse matrices and axis 0, so need to transpose back and
    # forth (cheap operation) to get my CSR with axis 1
    adj_stdev_1 = scale(adj_matrix.transpose().astype(float, copy=False),  # cast to float to avoid warning msg
                               axis=0, with_mean=False, with_std=True).transpose()

    row_means = adj_stdev_1.mean(axis=1)
    n = adj_matrix.shape[1]

    # old: score = (pair_x - row_means[row_idx1]).dot(pair_y - row_means[row_idx2]) / float(n)
    if sparse.isspmatrix(adj_matrix):
        transformed_mat = (adj_stdev_1 - row_means) / np.sqrt(n)
    else:
        transformed_mat = (adj_stdev_1 - row_means.reshape((-1,1))) / np.sqrt(n)

    return transformed_mat


