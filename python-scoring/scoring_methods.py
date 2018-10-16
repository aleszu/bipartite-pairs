import numpy as np
from scipy.spatial.distance import cosine
import pandas
from timeit import default_timer as timer
import scoring_methods_fast


all_defined_methods = ['jaccard', 'cosine', 'cosineIDF', 'sharedSize', 'hamming', 'pearson', 'weighted_corr',
                       'shared_weight11', 'shared_weight1100', 'adamic_adar', 'newman', 'mixed_pairs']
# Required args for methods:
# 'pi_vector' needed for cosineIDF, weighted_corr, shared_weight11, shared_weight1100, adamic_adar, newman
# 'num_docs' needed for adamic_adar and newman
# 'mixed_pairs_sims' needed for mixed_pairs

# Returns a table of scores with one column per method
# Direction of scores: higher for true pairs
def score_pairs(pairs_generator, generator_arg, which_methods, print_timing=False, **all_named_args):
    scores = {}

    # first pass through the generator to paste in row ids
    # (note: if this is slow, could build a separate generator to only give row ids)
    scores['item1'], scores['item2'] = item_ids(pairs_generator(generator_arg))

    if 'jaccard' in which_methods:
        scores['jaccard'] = compute_scores(pairs_generator(generator_arg), jaccard, print_timing=print_timing)
    if 'cosine' in which_methods:
        scores['cosine'] = compute_scores(pairs_generator(generator_arg), cosine_sim, print_timing=print_timing)
    if 'cosineIDF' in which_methods:
        idf_weights = np.log(1/all_named_args['pi_vector'])
        scores['cosineIDF'] = compute_scores(pairs_generator(generator_arg), cosine_sim, print_timing=print_timing,
                                             weights=idf_weights)
    if 'sharedSize' in which_methods:
        scores['sharedSize'] = compute_scores(pairs_generator(generator_arg), shared_size, print_timing=print_timing,
                                              back_compat=all_named_args.get('back_compat', False))
    if 'hamming' in which_methods:
        scores['hamming'] = compute_scores(pairs_generator(generator_arg), hamming, print_timing=print_timing,
                                           back_compat=all_named_args.get('back_compat', False))
    if 'pearson' in which_methods:
        scores['pearson'] = compute_scores(pairs_generator(generator_arg), pearson_cor, print_timing=print_timing)
    if 'weighted_corr' in which_methods:
        scores['weighted_corr'] = compute_scores(pairs_generator(generator_arg), weighted_corr, print_timing=print_timing,
                                                 p_i=all_named_args['pi_vector'])
        # scores['weighted_corr'] = scoring_methods_fast.simple_only_weighted_corr(pairs_generator, generator_arg,
        #                                                     pi_vector=all_named_args['pi_vector'],
        #                                                     print_timing=print_timing)['weighted_corr']
        # scores['weighted_corr'] = scoring_methods_fast.compute_scores_fast(pairs_generator, generator_arg,
        #                                                                   scoring_methods_fast.wc_transform,
        #                                                                   print_timing=print_timing, pi_vector=all_named_args['pi_vector'])

    if 'shared_weight11' in which_methods:
        scores['shared_weight11'] = compute_scores(pairs_generator(generator_arg), shared_weight11, print_timing=print_timing,
                                                 p_i=all_named_args['pi_vector'])
        # scores['shared_weight11'] = scoring_methods_fast.compute_scores_fast(pairs_generator, generator_arg,
        #                                                                   scoring_methods_fast.shared_weight11_transform,
        #                                                                   print_timing=print_timing,
        #                                                                   pi_vector=all_named_args['pi_vector'])
    if 'shared_weight1100' in which_methods:
        scores['shared_weight1100'] = compute_scores(pairs_generator(generator_arg), shared_weight1100, print_timing=print_timing,
                                                 p_i=all_named_args['pi_vector'])
    if 'adamic_adar' in which_methods:
        # for adamic_adar and newman, need to ensure every affil is seen at least twice (for the 1/1 terms,
        # which are all they use). this happens automatically if p_i was learned empirically. this keeps the score per
        # term in [0, 1].
        num_docs_word_occurs_in = np.maximum(all_named_args['num_docs'] * all_named_args['pi_vector'], 2)
        scores['adamic_adar'] = compute_scores(pairs_generator(generator_arg), adamic_adar, print_timing=print_timing,
                                          affil_counts=num_docs_word_occurs_in)
        # scores['adamic_adar'] = scoring_methods_fast.simple_only_adamic_adar_scores(pairs_generator, generator_arg,
        #                                                                              num_docs_word_occurs_in, print_timing=True)

        # slight syntax weirdness: 2nd arg must be adj_matrix, not just any old generator_arg
        # scores['adamic_adar'] = scoring_methods_fast.compute_scores_fast(pairs_generator, generator_arg,
        #                                                                   scoring_methods_fast.adamic_adar_transform,
        #                                                                   print_timing=print_timing, num_docs=all_named_args['num_docs'],
        #                                                                   pi_vector=all_named_args['pi_vector'])
    if 'newman' in which_methods:
        num_docs_word_occurs_in = np.maximum(all_named_args['num_docs'] * all_named_args['pi_vector'], 2)
        scores['newman'] = compute_scores(pairs_generator(generator_arg), newman, print_timing=print_timing,
                                          affil_counts=num_docs_word_occurs_in)
        # scores['newman'] = scoring_methods_fast.compute_scores_fast(pairs_generator, generator_arg,
        #                                                                   scoring_methods_fast.newman_transform,
        #                                                                   print_timing=print_timing, num_docs=all_named_args['num_docs'],
        #                                                                   pi_vector=all_named_args['pi_vector'])
    if 'mixed_pairs' in which_methods:
        for mp_sim in all_named_args['mixed_pairs_sims']:
            method_name = 'mixed_pairs_' + str(mp_sim)
            scores[method_name] = compute_scores(pairs_generator(generator_arg), mixed_pairs, print_timing=print_timing,
                                                   p_i = all_named_args['pi_vector'], sim=mp_sim)

    return pandas.DataFrame(scores)



# General method to return a vector of scores from running one method
def compute_scores(pairs_generator, sim_func, print_timing=False, **named_args_to_func):
    start = timer()
    scores = []
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator:
        scores.append(sim_func(pair_x, pair_y, **named_args_to_func))

    end = timer()
    print sim_func.__name__ + ": " + str(end - start) + " secs" if print_timing else ''

    return scores


def item_ids(pairs_generator):
    item1, item2 = [], []
    for (row_idx1, row_idx2, pair_x, pair_y) in pairs_generator:
        item1.append(row_idx1)
        item2.append(row_idx2)
    return(item1, item2)

# Jaccard, pearson, cosine and weighted cosine: avoid division by zero. Wherever we would see 0/0, return 0 instead.
def jaccard(x, y):
    #print ignored_args
    num_ones = np.logical_or(x, y).sum()  # cmpts where either vector has a 1
    if num_ones > 0:
        return float(np.logical_and(x, y).sum()) / num_ones
    else:
        return 0


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


def pearson_cor(x, y):
    if x.sum() == 0 or y.sum() == 0:
        return 0
    return np.corrcoef(x, y)[1, 0]  # np.corrcoef() gives a 2x2 matrix


# Normalized (going forward) to be in [0,1]
def hamming(x, y, back_compat = False):
    hd = np.logical_xor(x, y).sum()
    if back_compat:
        return hd
    else:
        return 1 - (float(hd)/x.shape[0])


# Normalized (going forward) to be in [0,1]
def shared_size(x, y, back_compat = False):
    m = np.logical_and(x, y).sum()
    if back_compat:
        return m
    else:
        return float(m) / x.shape[0]


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


def shared_weight11(x, y, p_i):
    return np.dot(np.log(1/p_i), np.logical_and(x, y))


def shared_weight1100(x, y, p_i):
    return np.dot(np.log(1/p_i), np.logical_and(x, y)) + \
            np.dot(np.log(1/(1-p_i)), np.logical_and(np.logical_not(x), np.logical_not(y)))

def adamic_adar(x, y, affil_counts):
    return np.dot(np.logical_and(x, y), 1/np.log(affil_counts))

def newman(x, y, affil_counts):
    return np.dot(np.logical_and(x, y), 1/(affil_counts.astype(float) - 1))


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

