import numpy as np
import pandas
from scipy import sparse
from timeit import default_timer as timer
import scoring_methods_fast
import extra_implementations

all_defined_methods = ['jaccard', 'cosine', 'cosineIDF', 'shared_size', 'hamming', 'pearson', 'weighted_corr',
                       'shared_weight11', 'shared_weight1100', 'adamic_adar', 'newman', 'mixed_pairs']
# Required args for methods:
# 'pi_vector' needed for cosineIDF, weighted_corr, shared_weight11, shared_weight1100, adamic_adar, newman
# 'num_docs' needed for adamic_adar and newman
# 'mixed_pairs_sims' needed for mixed_pairs

# pairs_generator: a function that takes adj_matrix as an argument
# run_all_implementations: can use False (default), True or 1 for more, or 2 to do even the slow ones
# Returns a table of scores with one column per method
# Direction of returned scores: higher for true pairs
def score_pairs(pairs_generator, adj_matrix, which_methods, print_timing=False,
                run_all_implementations=False, **all_named_args):
    scores = {}

    # first pass through the generator to paste in row ids
    scores['item1'], scores['item2'] = item_ids(pairs_generator(adj_matrix))

    if 'jaccard' in which_methods:
        scores['jaccard'] = compute_scores_orig(pairs_generator(adj_matrix), jaccard, print_timing=print_timing)
    if 'cosine' in which_methods:
        scores['cosine'] = scoring_methods_fast.simple_only_cosine(pairs_generator, adj_matrix,
                                                                   print_timing=print_timing, use_package=True)
        if run_all_implementations >= 2:
            scores['cosine'] = compute_scores_orig(pairs_generator(adj_matrix), extra_implementations.cosine_sim,
                                                   print_timing=print_timing)

            scores['cosine'] = scoring_methods_fast.simple_only_cosine(pairs_generator, adj_matrix,
                                                                       print_timing=print_timing, use_package=False)

    if 'cosineIDF' in which_methods:
        idf_weights = np.log(1/all_named_args['pi_vector'])
        scores['cosineIDF'] = scoring_methods_fast.simple_only_cosine(pairs_generator, adj_matrix, weights=idf_weights,
                                                                      print_timing=print_timing, use_package=True)
        if run_all_implementations >= 2:
            scores['cosineIDF'] = compute_scores_orig(pairs_generator(adj_matrix), extra_implementations.cosine_sim,
                                                      print_timing=print_timing, weights=idf_weights)
            scores['cosineIDF'] = scoring_methods_fast.simple_only_cosine(pairs_generator, adj_matrix, weights=idf_weights,
                                                                          print_timing=print_timing, use_package=False)

    if 'shared_size' in which_methods:
        scores['shared_size'] = compute_scores_with_transform(pairs_generator, adj_matrix,
                                                              shared_size_transform,
                                                              print_timing=print_timing,
                                                              back_compat=all_named_args.get('back_compat', False))
        if run_all_implementations >= 2:
            scores['shared_size'] = compute_scores_orig(pairs_generator(adj_matrix), extra_implementations.shared_size,
                                                       print_timing=print_timing,
                                                       back_compat=all_named_args.get('back_compat', False))

    if 'hamming' in which_methods:
        scores['hamming'] = compute_scores_orig(pairs_generator(adj_matrix), hamming, print_timing=print_timing,
                                                back_compat=all_named_args.get('back_compat', False))
    if 'pearson' in which_methods:
        scores['pearson'] = scoring_methods_fast.simple_only_pearson(pairs_generator, adj_matrix,
                                                                      print_timing=print_timing)
        if run_all_implementations:
            scores['pearson'] = extra_implementations.simple_only_phi_coeff(pairs_generator, adj_matrix,
                                                                        print_timing=print_timing)
        if run_all_implementations >= 2:
            scores['pearson'] = compute_scores_orig(pairs_generator(adj_matrix), extra_implementations.pearson_as_phi,
                                                    print_timing=print_timing)
            scores['pearson'] = compute_scores_orig(pairs_generator(adj_matrix), extra_implementations.pearson_cor, print_timing=print_timing)

    if 'weighted_corr' in which_methods:
        if sparse.isspmatrix(adj_matrix):
            # keep it sparse; can't use fastest method
            scores['weighted_corr'] = compute_scores_from_terms(pairs_generator, adj_matrix, wc_terms,
                                                                pi_vector=all_named_args['pi_vector'],
                                                                num_affils=adj_matrix.shape[1], print_timing=print_timing)

        else:
            scores['weighted_corr'] = compute_scores_with_transform(pairs_generator, adj_matrix, wc_transform,
                                                                    print_timing=print_timing,
                                                                    pi_vector=all_named_args['pi_vector'])

        if run_all_implementations >= 2:
            scores['weighted_corr'] = compute_scores_orig(pairs_generator(adj_matrix), extra_implementations.weighted_corr,
                                                          print_timing=print_timing, p_i=all_named_args['pi_vector'])
            scores['weighted_corr'] = extra_implementations.compute_scores_from_terms0(pairs_generator, adj_matrix,
                                                                wc_terms, pi_vector=all_named_args['pi_vector'],
                                                                num_affils=adj_matrix.shape[1], print_timing=print_timing)
        if run_all_implementations:
            scores['weighted_corr'] = extra_implementations.simple_weighted_corr_sparse(pairs_generator, adj_matrix,
                                                                                       pi_vector=all_named_args['pi_vector'],
                                                                                       print_timing=print_timing)
            if not sparse.isspmatrix(adj_matrix):
                scores['weighted_corr'] = extra_implementations.simple_only_weighted_corr(
                                            pairs_generator, adj_matrix, pi_vector=all_named_args['pi_vector'],
                                            print_timing=print_timing)['weighted_corr']

    if 'shared_weight11' in which_methods:
        scores['shared_weight11'] = compute_scores_with_transform(pairs_generator, adj_matrix,
                                                                  shared_weight11_transform,
                                                                  print_timing=print_timing,
                                                                  pi_vector=all_named_args['pi_vector'])
        if run_all_implementations >= 2:
            scores['shared_weight11'] = compute_scores_orig(pairs_generator(adj_matrix),
                                                            extra_implementations.shared_weight11, print_timing=print_timing,
                                                            p_i=all_named_args['pi_vector'])
    if 'shared_weight1100' in which_methods:
        scores['shared_weight1100'] = compute_scores_from_terms(pairs_generator, adj_matrix, shared_weight1100_terms,
                                                                pi_vector=all_named_args['pi_vector'],
                                                                print_timing=print_timing)
        if run_all_implementations:
            scores['shared_weight1100'] = extra_implementations.compute_scores_from_terms0(
                pairs_generator, adj_matrix, shared_weight1100_terms,
                pi_vector=all_named_args['pi_vector'], print_timing=print_timing)
        if run_all_implementations >= 2:
            scores['shared_weight1100'] = compute_scores_orig(pairs_generator(adj_matrix), extra_implementations.shared_weight1100,
                                                              print_timing=print_timing, p_i=all_named_args['pi_vector'])

    if 'adamic_adar' in which_methods:
        # for adamic_adar and newman, need to ensure every affil is seen at least twice (for the 1/1 terms,
        # which are all they use). this happens automatically if p_i was learned empirically. this keeps the score per
        # term in [0, 1].
        num_docs_word_occurs_in = np.maximum(all_named_args['num_docs'] * all_named_args['pi_vector'], 2)
        scores['adamic_adar'] = scoring_methods_fast.simple_only_adamic_adar_scores(pairs_generator, adj_matrix,
                                                                                     num_docs_word_occurs_in,
                                                                                     print_timing=True)

        if run_all_implementations:
            scores['adamic_adar'] = compute_scores_with_transform(pairs_generator, adj_matrix,
                                                              adamic_adar_transform,
                                                              print_timing=print_timing,
                                                              num_docs=all_named_args['num_docs'],
                                                              pi_vector=all_named_args['pi_vector'])
        if run_all_implementations >= 2:
            scores['adamic_adar'] = compute_scores_orig(pairs_generator(adj_matrix), extra_implementations.adamic_adar,
                                                        print_timing=print_timing, affil_counts=num_docs_word_occurs_in)
    if 'newman' in which_methods:
        num_docs_word_occurs_in = np.maximum(all_named_args['num_docs'] * all_named_args['pi_vector'], 2)
        scores['newman'] = compute_scores_with_transform(pairs_generator, adj_matrix,
                                                         newman_transform,
                                                         print_timing=print_timing, num_docs=all_named_args['num_docs'],
                                                         pi_vector=all_named_args['pi_vector'])
        if run_all_implementations >= 2:
            scores['newman'] = compute_scores_orig(pairs_generator(adj_matrix), extra_implementations.newman,
                                                   print_timing=print_timing, affil_counts=num_docs_word_occurs_in)
    if 'mixed_pairs' in which_methods:
        for mp_sim in all_named_args['mixed_pairs_sims']:
            method_name = 'mixed_pairs_' + str(mp_sim)
            scores[method_name] = compute_scores_from_terms(pairs_generator, adj_matrix, mixed_pairs_terms,
                                                            pi_vector=all_named_args['pi_vector'], sim=mp_sim,
                                                            print_timing=print_timing)
            if run_all_implementations:
                scores[method_name] = extra_implementations.compute_scores_from_terms0(pairs_generator, adj_matrix, mixed_pairs_terms,
                                                            pi_vector=all_named_args['pi_vector'], sim=mp_sim, print_timing=print_timing)
            if run_all_implementations >= 2:
                scores[method_name] = compute_scores_orig(pairs_generator(adj_matrix), extra_implementations.mixed_pairs,
                                                      p_i = all_named_args['pi_vector'], sim=mp_sim,
                                                      print_timing=print_timing)

    return pandas.DataFrame(scores)


def item_ids(pairs_generator):
    item1, item2 = [], []
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator:
        item1.append(row_idx1)
        item2.append(row_idx2)
    return(item1, item2)


# General method to return a vector of scores from running one method
def compute_scores_orig(pairs_generator, sim_func, print_timing=False, **named_args_to_func):
    start = timer()
    scores = []
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator:
        scores.append(sim_func(pair_x, pair_y, **named_args_to_func))

    end = timer()
    if print_timing:
        print "original " + sim_func.__name__ + ": " + str(end - start) + " secs"

    return scores


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



# This version: rather than explicitly computing the scores for 1/0 terms, have a base_score that assume all entries
# are 1/0s, and adjust it for 11 and 00s.
def compute_scores_from_terms(pairs_generator, adj_matrix, scores_bi_func, print_timing=False, **named_args_to_func):
    start = timer()
    terms_for_11, value_10, terms_for_00 = scores_bi_func(**named_args_to_func)
    scores = []
    base_score = value_10 * adj_matrix.shape[1]
    terms_for_11 -= value_10
    terms_for_00 -= value_10
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(adj_matrix):
        sum_11 = terms_for_11.dot(pair_x * pair_y)
        sum_00 = terms_for_00.dot(np.logical_not(pair_x) * np.logical_not(pair_y))
        scores.append(base_score + sum_11 + sum_00 )

    end = timer()
    if print_timing:
        print scores_bi_func.__name__ + ": " + str(end - start) + " secs"
    return scores


def shared_weight1100_terms(pi_vector):
    terms_for_11 = np.log(1/pi_vector)
    value_10 = 0
    terms_for_00 = np.log(1/(1-pi_vector))
    return terms_for_11, value_10, terms_for_00

def mixed_pairs_terms(pi_vector, sim):
    terms_for_11 = np.log((sim + (1 - sim) * pi_vector) / pi_vector)
    value_10 = np.log(1 - sim)
    terms_for_00 = np.log((1 - pi_vector + sim * pi_vector)/(1 - pi_vector))
    return terms_for_11, value_10, terms_for_00

def wc_terms(pi_vector, num_affils):
    terms_for_11 = (1 - pi_vector) / (num_affils * pi_vector)
    value_10 = -1 / float(num_affils)
    terms_for_00 = pi_vector / ((1 - pi_vector) * num_affils)
    return terms_for_11, value_10, terms_for_00



# Turns out the scoring functions can be ~10x faster when the function is written as: transform the adjacency matrix, then
# take dot products of the row pairs.
def compute_scores_with_transform(pairs_generator, adj_matrix, transf_func, print_timing=False, **named_args_to_func):
    start = timer()
    transformed_mat = transf_func(adj_matrix, **named_args_to_func)
    scores = []
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator(transformed_mat):
        scores.append(pair_x.dot(pair_y))
    end = timer()
    if print_timing:
        print transf_func.__name__ + ": " + str(end - start) + " secs"
    return scores

# Standardize adj_matrix column-wise: for each coln, x --> (x - p_i) / sqrt(p_i(1-p_i))
# Also multiply denominator by sqrt(num_cols), so that the final dot product returns the mean, not just a sum.
# Necessarily makes a dense matrix.
def wc_transform(adj_matrix, pi_vector):
    return (adj_matrix - pi_vector) / np.sqrt(pi_vector * (1 - pi_vector) * adj_matrix.shape[1])

# Leaves matrix sparse if it starts sparse
def newman_transform(adj_matrix, pi_vector, num_docs):
    affil_counts = np.maximum(num_docs * pi_vector, 2)
    if sparse.isspmatrix(adj_matrix):
        return adj_matrix.multiply(1/np.sqrt(affil_counts.astype(float) - 1)).tocsr()
    else:
        return adj_matrix / np.sqrt(affil_counts.astype(float) - 1)

# Leaves matrix sparse if it starts sparse
def shared_size_transform(adj_matrix, back_compat = False):
    if back_compat:
        return adj_matrix
    else:  # todo: test this version
        if sparse.isspmatrix(adj_matrix):
            return adj_matrix.multiply(np.sqrt(adj_matrix.shape[1]))
        else:
            return adj_matrix * np.sqrt(adj_matrix.shape[1])

# Leaves matrix sparse if it starts sparse
def shared_weight11_transform(adj_matrix, pi_vector):
    if sparse.isspmatrix(adj_matrix):
        # Keep the matrix sparse if it was before. (By default changes to coo() if I don't cast it tocsr().)
        return adj_matrix.multiply(np.sqrt(np.log(1 / pi_vector))).tocsr()  # .multiply() doesn't exist if adj_matrix is dense
    else:
        return adj_matrix * np.sqrt(np.log(1 / pi_vector))  # gives incorrect behavior if adj_matrix is sparse


# Leaves matrix sparse if it starts sparse
def adamic_adar_transform(adj_matrix, pi_vector, num_docs):
    affil_counts = np.maximum(num_docs * pi_vector, 2)
    if sparse.isspmatrix(adj_matrix):
        return adj_matrix.multiply(1/np.sqrt(np.log(affil_counts))).tocsr()
    else:
        return adj_matrix / np.sqrt(np.log(affil_counts))



# Current status on speed (notes to self):
# -Dense matrix calcs are much faster. It's a time vs. space tradeoff.
# -Slowest methods, when using sparse adj_mat:
#   weightedCorr, shared_weight1100, mixedPairs; jaccard
# -Slowest methods when using dense adj_mat:
#   mixedPairs, shared_weight1100; jaccard; hamming, pearson
#       --> The *_terms methods are the slowest. (Better than the original implementations, but still not very efficient.)
# -when calling score_pairs with test_all_versions=True, it uses the one that's been fastest on the example data,
#  but best one could change on different sized data sets, and depending on whether matrix is sparse or dense.

