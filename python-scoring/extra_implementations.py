from __future__ import print_function
# Scoring functions for which I've found a faster implementation. Keeping these around for
# clarity and in case I want them back.

from builtins import str
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from timeit import default_timer as timer
import pandas
import transforms_for_dot_prods
import scoring_methods
import scoring_methods_fast

# Instead of clogging up score_pairs with these slower versions, put the calls together and out of the way.
# This function runs those that are inside the "if run_all_implementations >= 2:" branches and others that
# are definitely slower than the main versions.
def run_extra_implementations2(pairs_generator, adj_matrix, which_methods, scores_storage_magic_d,
                               print_timing=False, **all_named_args):
    if which_methods == 'all':
        which_methods = scoring_methods.all_defined_methods
    if all_named_args.get('mixed_pairs_sims', None) == 'standard':
        all_named_args['mixed_pairs_sims'] = (.1, .01, .001)


    if 'cosine' in which_methods:
        out_data = scores_storage_magic_d.create_and_store_array('cosine')
        compute_scores_orig(pairs_generator(adj_matrix), cosine_sim, out_data,
                            print_timing=print_timing)
        scoring_methods_fast.simple_only_cosine(pairs_generator, adj_matrix, out_data,
                                                print_timing=print_timing, use_package=False)
    if 'cosineIDF' in which_methods:
        idf_weights = np.log(1/all_named_args['pi_vector'])
        out_data = scores_storage_magic_d.create_and_store_array('cosineIDF')
        compute_scores_orig(pairs_generator(adj_matrix), cosine_sim, out_data,
                            print_timing=print_timing, weights=idf_weights)
        scoring_methods_fast.simple_only_cosine(pairs_generator, adj_matrix, out_data, weights=idf_weights,
                                                print_timing=print_timing, use_package=False)
    if 'shared_size' in which_methods:
        back_compat = all_named_args.get('back_compat', False)
        out_data = scores_storage_magic_d.create_and_store_array('shared_size', dtype = int if back_compat else float)
        compute_scores_orig(pairs_generator(adj_matrix), shared_size, out_data,
                            print_timing=print_timing,
                            back_compat=all_named_args.get('back_compat', False))
    if 'adamic_adar' in which_methods:
        # for adamic_adar and newman, need to ensure every affil is seen at least twice (for the 1/1 terms,
        # which are all they use). this happens automatically if p_i was learned empirically. this keeps the score per
        # term in [0, 1].
        num_docs_word_occurs_in = np.maximum(all_named_args['num_docs'] * all_named_args['pi_vector'], 2)
        out_data = scores_storage_magic_d.create_and_store_array('adamic_adar')
        compute_scores_orig(pairs_generator(adj_matrix), adamic_adar, out_data,
                            print_timing=print_timing, affil_counts=num_docs_word_occurs_in)
    if 'newman' in which_methods:
        num_docs_word_occurs_in = np.maximum(all_named_args['num_docs'] * all_named_args['pi_vector'], 2)
        out_data = scores_storage_magic_d.create_and_store_array('newman')
        compute_scores_orig(pairs_generator(adj_matrix), newman, out_data,
                            print_timing=print_timing, affil_counts=num_docs_word_occurs_in)

    if 'shared_weight11' in which_methods:
        out_data = scores_storage_magic_d.create_and_store_array('shared_weight11')
        compute_scores_orig(pairs_generator(adj_matrix), shared_weight11, out_data,
                            p_i=all_named_args['pi_vector'], print_timing=print_timing)

    if 'pearson' in which_methods:
        out_data = scores_storage_magic_d.create_and_store_array('pearson')
        simple_only_phi_coeff(pairs_generator, adj_matrix, out_data, print_timing=print_timing)
        compute_scores_orig(pairs_generator(adj_matrix), pearson_as_phi, out_data, print_timing=print_timing)
        compute_scores_orig(pairs_generator(adj_matrix), pearson_cor, out_data, print_timing=print_timing)
    if 'hamming' in which_methods:
        back_compat = all_named_args.get('back_compat', False)
        out_data = scores_storage_magic_d.create_and_store_array('hamming', dtype=int if back_compat else float)
        compute_scores_orig(pairs_generator(adj_matrix), hamming, out_data, print_timing=print_timing,
                            back_compat=back_compat)
    if 'jaccard' in which_methods:
        out_data = scores_storage_magic_d.create_and_store_array('jaccard')
        compute_scores_orig(pairs_generator(adj_matrix), jaccard, out_data, print_timing=print_timing)
    if 'weighted_corr' in which_methods:
        out_data = scores_storage_magic_d.create_and_store_array('weighted_corr')
        compute_scores_orig(pairs_generator(adj_matrix), weighted_corr, out_data,
                            print_timing=print_timing, p_i=all_named_args['pi_vector'])
        compute_scores_from_terms0(pairs_generator, adj_matrix, scoring_methods.wc_terms, out_data,
                                   pi_vector=all_named_args['pi_vector'], num_affils=adj_matrix.shape[1],
                                   print_timing=print_timing)

    if 'shared_weight1100' in which_methods:
        out_data = scores_storage_magic_d.create_and_store_array('shared_weight1100')
        compute_scores_from_terms0(pairs_generator, adj_matrix, scoring_methods.shared_weight1100_terms, out_data,
                                   pi_vector=all_named_args['pi_vector'], print_timing=print_timing)

        compute_scores_orig(pairs_generator(adj_matrix), shared_weight1100, out_data,
                            print_timing=print_timing, p_i=all_named_args['pi_vector'])

    if 'mixed_pairs' in which_methods:
        for mp_sim in all_named_args['mixed_pairs_sims']:
            method_name = 'mixed_pairs_' + str(mp_sim)
            out_data = scores_storage_magic_d.create_and_store_array(method_name)
            compute_scores_orig(pairs_generator(adj_matrix), mixed_pairs, out_data,
                                p_i = all_named_args['pi_vector'], sim=mp_sim, print_timing=print_timing)
            compute_scores_from_terms0(pairs_generator, adj_matrix, scoring_methods.mixed_pairs_terms, out_data,
                                       pi_vector=all_named_args['pi_vector'], sim=mp_sim, print_timing=print_timing)



# General method to return a vector of scores from running one method
def compute_scores_orig(pairs_generator, sim_func, scores_out, print_timing=False, **named_args_to_func):
    start = timer()
    for (i, j, _, _, pair_x, pair_y) in pairs_generator:
        scores_out[i,j] = (sim_func(pair_x, pair_y, **named_args_to_func))

    end = timer()
    if print_timing:
        print("original " + sim_func.__name__ + ": " + str(end - start) + " secs")


## Indiv pair computations that can be called from compute_scores_orig ##

# Jaccard, pearson, cosine and weighted cosine: avoid division by zero. Wherever we would see 0/0, return 0 instead.
def jaccard(x, y):
    num_ones = np.logical_or(x, y).sum()  # cmpts where either vector has a 1
    if num_ones > 0:
        return float(x.dot(y)) / num_ones
    else:
        return 0


# Normalized (going forward) to be in [0,1]
def hamming(x, y, back_compat = False):
    hd = np.logical_xor(x, y).sum()
    if back_compat:
        return hd
    else:
        return 1 - (float(hd)/x.shape[0])


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




def compute_scores_from_terms0(pairs_generator, adj_matrix, scores_bi_func, scores_out,
                               print_timing=False, **named_args_to_func):
    start = timer()
    terms_for_11, value_10, terms_for_00 = scores_bi_func(**named_args_to_func)
    # scores = []
    for (i, j, _, _, pair_x, pair_y) in pairs_generator(adj_matrix):
        sum_11 = terms_for_11.dot(pair_x * pair_y)
        sum_00 = terms_for_00.dot(np.logical_not(pair_x) * np.logical_not(pair_y))
        sum_10 = value_10 * np.logical_xor(pair_x, pair_y).sum()
        scores_out[i,j] = (sum_11 + sum_10 + sum_00)

    end = timer()
    if print_timing:
        print(scores_bi_func.__name__ + ": " + str(end - start) + " secs")
    # return scores

## Helpers to be called with compute_scores_from_terms or compute_scores_from_terms0  ##



## Helpers to be called with compute_scores_from_transform ##




## Single-purpose functions ##

# Leaves matrix sparse if it starts sparse
def simple_only_phi_coeff(pairs_generator, adj_matrix, scores_out, print_timing=False):
    start = timer()

    row_sums = adj_matrix.sum(axis=1)
    if type(row_sums) == np.matrix:
        row_sums = row_sums.A1
    n = adj_matrix.shape[1]
    # scores = []
    for (row_idx1, row_idx2, _, _, pair_x, pair_y) in pairs_generator(adj_matrix):
        n11 = np.logical_and(pair_x, pair_y).sum()
        scores_out[row_idx1, row_idx2] = ( (n * n11 - row_sums[row_idx1] * row_sums[row_idx2]) /
                 np.sqrt(row_sums[row_idx1] * row_sums[row_idx2] * (n - row_sums[row_idx1]) * (n - row_sums[row_idx2])) )

    end = timer()
    if print_timing:
        print('simple_only_phi_coeff: ' + str(end - start) + " secs")
    # return scores



# Necessarily makes a dense matrix.
def simple_only_weighted_corr(pairs_generator, adj_matrix, scores_out, pi_vector, print_timing=False):
    start = timer()
    transformed_mat = transforms_for_dot_prods.wc_transform(adj_matrix, pi_vector)

    # item1, item2, wc = [], [], []
    for (row_idx1, row_idx2, item1_id, item2_id, pair_x, pair_y) in pairs_generator(transformed_mat):
        # item1.append(item1_id)
        # item2.append(item2_id)
        scores_out[row_idx1, row_idx2] = (pair_x.dot(pair_y))

    end = timer()
    if print_timing:
        print('simple_only_weighted_corr: ' + str(end - start) + " secs")
    # return pandas.DataFrame({'item1': item1, 'item2': item2, 'weighted_corr': wc})


# Leaves matrix sparse if it starts sparse
def simple_weighted_corr_sparse(pairs_generator, adj_matrix, scores_out, pi_vector, print_timing=False):
    start = timer()

    terms_for_11 = (1 - pi_vector) / pi_vector
    #sqrt_terms_for_10 = np.full_like(pi_vector, fill_value=-1)
    terms_for_00 = pi_vector / (1 - pi_vector)

    # wc = []
    n = float(adj_matrix.shape[1])
    for (i, j, _, _, pair_x, pair_y) in pairs_generator(adj_matrix):
        sum_11 = (pair_x * pair_y).dot(terms_for_11)
        sum_00 = (np.logical_not(pair_x) * np.logical_not(pair_y)).dot(terms_for_00)
        sum_10 = -np.logical_xor(pair_x, pair_y).sum()
        scores_out[i,j] = ((sum_11 + sum_10 + sum_00) / n)

    end = timer()
    if print_timing:
        print('simple_weighted_corr_sparse: ' + str(end - start) + " secs")
    # return wc
