# Scoring functions for which I've found a faster implementation. Keeping these around for
# clarity and in case I want them back.

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from timeit import default_timer as timer
import pandas
import scoring_methods

#import scoring_methods

## Indiv pair computations that can be called from compute_scores_orig ##

def weighted_corr(x, y, p_i):

    # score(1/1) = (1-p)/p
    # score(1/0) = -1
    # score(0/0) = p/(1-p)
    terms_for_11 = (1 - p_i) / p_i
    terms_for_10 = np.full_like(p_i, fill_value=-1)
    terms_for_00 = p_i / (1 - p_i)

    positions_10 = np.logical_xor(x, y)
    positions_11 = np.logical_and(x, y)
    positions_00 = np.logical_and(np.logical_not(x), np.logical_not(y))

    return (np.dot(positions_11, terms_for_11) + np.dot(positions_10, terms_for_10) +
            np.dot(positions_00, terms_for_00)) / float(p_i.shape[0])

    # alt method, might be slower
    # full_vector = terms_for_00
    # full_vector[positions_10] = terms_for_10[positions_10]
    # full_vector[positions_11] = terms_for_11[positions_11]
    #
    # return np.mean(full_vector)



def mixed_pairs(x, y, p_i, sim):
    # score(1/1) = log((s + (1-s)p_i) / p_i)
    # score(1/0) = log((1 - p_i + p_i*s)/(1-p_i))
    # score(0/0) = log(1 - s)
    terms_for_11 = np.log((sim + (1 - sim) * p_i) / p_i)
    terms_for_00 = np.log((1 - p_i + sim * p_i)/(1 - p_i))
    terms_for_10 = np.full_like(p_i, fill_value=np.log(1 - sim))

    positions_10 = np.logical_xor(x, y)
    positions_11 = np.logical_and(x, y)
    positions_00 = np.logical_and(np.logical_not(x), np.logical_not(y))

    return (np.dot(positions_11, terms_for_11) + np.dot(positions_10, terms_for_10) +
            np.dot(positions_00, terms_for_00))


def pearson_cor(x, y):
    if x.sum() == 0 or y.sum() == 0:
        return 0
    return pearsonr(x, y)[0]        # slightly faster than np.corrcoef(), far faster than sklearn.metrics.matthews_corrcoef()

# Faster than pearson_cor
def pearson_as_phi(x, y):
    if x.sum() == 0 or y.sum() == 0:
        return 0
    # phi = (n n11 [both] - n1. n.1 [each marginal]) /
    #           sqrt(n1. n.1 (n - n1.) (n - n.1) )
    n = len(x)
    n11 = x.dot(y)    # same as np.logical_and(x, y).sum()
    nx = x.sum()
    ny = y.sum()
    return (n * n11 - nx * ny) / np.sqrt(nx * ny * (n - nx) * (n - ny))


# slow, but sklearn.metrics.pairwise.cosine_similarity was worse
def cosine_sim(x, y, **named_args):
    if x.sum() == 0 or y.sum() == 0:
        return 0
    if named_args is not None and 'weights' in named_args:
        xw = x * named_args['weights']
        yw = y * named_args['weights']
    else:
        xw = x
        yw = y

    return 1 - cosine(xw, yw)  # spatial.distance.cosine gives (1 - the.usual.cosine)


# Normalized (going forward) to be in [0,1]
def shared_size(x, y, back_compat = False):
    m = x.dot(y)
    if back_compat:
        return m
    else:
        return float(m) / x.shape[0]


def shared_weight11(x, y, p_i):
    return np.dot(np.log(1/p_i), np.logical_and(x, y))


def shared_weight1100(x, y, p_i):
    return np.dot(np.log(1/p_i), np.logical_and(x, y)) + \
            np.dot(np.log(1/(1-p_i)), np.logical_and(np.logical_not(x), np.logical_not(y)))


def adamic_adar(x, y, affil_counts):
    return np.dot(np.logical_and(x, y), 1/np.log(affil_counts))


def newman(x, y, affil_counts):
    return np.dot(np.logical_and(x, y), 1/(affil_counts.astype(float) - 1))




def compute_scores_from_terms0(pairs_generator, adj_matrix, scores_bi_func, print_timing=False, **named_args_to_func):
    start = timer()
    terms_for_11, value_10, terms_for_00 = scores_bi_func(**named_args_to_func)
    scores = []
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(adj_matrix):
        sum_11 = terms_for_11.dot(pair_x * pair_y)
        sum_00 = terms_for_00.dot(np.logical_not(pair_x) * np.logical_not(pair_y))
        sum_10 = value_10 * np.logical_xor(pair_x, pair_y).sum()
        scores.append(sum_11 + sum_10 + sum_00)

    end = timer()
    if print_timing:
        print scores_bi_func.__name__ + ": " + str(end - start) + " secs"
    return scores

## Helpers to be called with compute_scores_from_terms or compute_scores_from_terms0  ##



## Helpers to be called with compute_scores_from_transform ##




## Single-purpose functions ##

# Leaves matrix sparse if it starts sparse
def simple_only_phi_coeff(pairs_generator, adj_matrix, print_timing=False):
    start = timer()

    row_sums = adj_matrix.sum(axis=1)
    if type(row_sums) == np.matrix:
        row_sums = row_sums.A1
    n = adj_matrix.shape[1]
    scores = []
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(adj_matrix):
        n11 = np.logical_and(pair_x, pair_y).sum()
        scores.append( (n * n11 - row_sums[row_idx1] * row_sums[row_idx2]) /
                 np.sqrt(row_sums[row_idx1] * row_sums[row_idx2] * (n - row_sums[row_idx1]) * (n - row_sums[row_idx2])) )

    end = timer()
    if print_timing:
        print 'simple_only_phi_coeff: ' + str(end - start) + " secs"
    return scores



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
    if print_timing:
        print 'simple_only_weighted_corr: ' + str(end - start) + " secs"
    return pandas.DataFrame({'item1': item1, 'item2': item2, 'weighted_corr': wc})


# Leaves matrix sparse if it starts sparse
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
    if print_timing:
        print 'simple_weighted_corr_sparse: ' + str(end - start) + " secs"
    return wc
