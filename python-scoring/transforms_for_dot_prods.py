from scipy import sparse
import numpy as np

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
        return adj_matrix.multiply(1/np.sqrt(affil_counts.astype(float) - 1)).tocsr()
    else:
        return adj_matrix / np.sqrt(affil_counts.astype(float) - 1)

# Leaves matrix sparse if it starts sparse
def shared_size_transform(adj_matrix, back_compat = False, make_dense=False):
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
        return adj_matrix.multiply(np.sqrt(np.log(1 / pi_vector))).tocsr()  # .multiply() doesn't exist if adj_matrix is dense
    else:
        return adj_matrix * np.sqrt(np.log(1 / pi_vector))  # gives incorrect behavior if adj_matrix is sparse


# Leaves matrix sparse if it starts sparse
def adamic_adar_transform(adj_matrix, pi_vector, num_docs, make_dense=False):
    affil_counts = np.maximum(num_docs * pi_vector, 2)
    if sparse.isspmatrix(adj_matrix) and not make_dense:
        return adj_matrix.multiply(1/np.sqrt(np.log(affil_counts))).tocsr()
    else:
        return adj_matrix / np.sqrt(np.log(affil_counts))

