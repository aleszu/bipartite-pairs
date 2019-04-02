from __future__ import print_function
from builtins import str
import numpy as np
from scipy import sparse
from timeit import default_timer as timer
#from pympler import asizeof
import scoring_methods_fast
import extra_implementations
import transforms_for_dot_prods
import magic_dictionary

all_defined_methods = ['jaccard', 'cosine', 'cosineIDF', 'shared_size', 'hamming', 'pearson',
                       'shared_weight11', 'shared_weight1100', 'adamic_adar', 'newman', 'mixed_pairs',
                       'weighted_corr', 'weighted_corr_exp']

def score_pairs(pairs_generator, adj_matrix, which_methods, outfile_csv_gz=None, print_timing=False,
                run_all_implementations=False, prefer_faiss=False, **all_named_args):
    """
    :param pairs_generator: a function that takes adj_matrix as an argument
    :param adj_matrix:
    :param which_methods:
    :param outfile_csv_gz:
    :param print_timing:
    :param run_all_implementations: can use False (default), True or 1 for more, or 2 to do even the slow ones
    :param prefer_faiss:
    :param all_named_args:
        Required args for methods:
        'pi_vector' needed for cosineIDF, weighted_corr, shared_weight11, shared_weight1100, adamic_adar, newman
        'num_docs' needed for adamic_adar and newman
        'mixed_pairs_sims' needed for mixed_pairs
        'exp_model' needed for wc_exp
    :return: Returns a table of scores with one column per method.
    Direction of returned scores: higher for true pairs.
    """

    if which_methods == 'all':
        which_methods = all_defined_methods
    if all_named_args.get('mixed_pairs_sims', None) == 'standard':
        all_named_args['mixed_pairs_sims'] = (.1, .01, .001)

    scores_storage = magic_dictionary.make_me_a_dict(adj_matrix.shape[0], force_memmap=True)

    which_methods, methods_for_faiss = separate_faiss_methods(which_methods, prefer_faiss, sparse.isspmatrix(adj_matrix))
    if len(methods_for_faiss):
        import scoring_with_faiss
        scoring_with_faiss.score_pairs_faiss_all_exact(adj_matrix, methods_for_faiss, scores_storage,
                                                                      print_timing=print_timing, **all_named_args)
        for oldcol in scores_storage.getkeys():
            if oldcol[-6:] == '_faiss':
                scores_storage.rename_method(oldcol, oldcol[:-6])
            if oldcol[:17] == 'mixed_pairs_faiss':
                scores_storage.rename_method(oldcol, 'mixed_pairs' + oldcol[17:])

    # Run them with fastest first:
    # 'cosine', 'cosineIDF'
    #'shared_size', 'adamic_adar', 'newman', 'shared_weight11'   -- use fast "transform"
    #'weighted_corr' and 'weighted_corr_exp' use "transform" when dense, "terms" when sparse -- speed varies accordingly
    #'shared_weight1100', 'mixed_pairs' -- only have "terms" method, uses 2 calls to transform when dense
    #'hamming', 'jaccard'  -- now fast if shared_size done first
    #'pearson'   -- currently the slowest

    if 'cosine' in which_methods:
        out_data = scores_storage.create_and_store_array('cosine')
        scoring_methods_fast.simple_only_cosine(pairs_generator, adj_matrix, out_data,
                                                print_timing=print_timing, use_package=True,
                                                use_batches=True)

    if 'cosineIDF' in which_methods:
        idf_weights = np.log(1/all_named_args['pi_vector'])
        out_data = scores_storage.create_and_store_array('cosineIDF')
        scoring_methods_fast.simple_only_cosine(pairs_generator, adj_matrix, out_data, weights=idf_weights,
                                                print_timing=print_timing, use_package=True,
                                                use_batches=True)

    if 'shared_size' in which_methods:
        back_compat = all_named_args.get('back_compat', False)
        out_data = scores_storage.create_and_store_array('shared_size', dtype=int if back_compat else float)
        compute_scores_with_transform(pairs_generator, adj_matrix,
                                      transforms_for_dot_prods.shared_size_transform, out_data,
                                      print_timing=print_timing,
                                      back_compat=all_named_args.get('back_compat', False))
    if 'adamic_adar' in which_methods:
        out_data = scores_storage.create_and_store_array('adamic_adar')
        compute_scores_with_transform(pairs_generator, adj_matrix,
                                      transforms_for_dot_prods.adamic_adar_transform, out_data,
                                      print_timing=print_timing,
                                      num_docs=all_named_args['num_docs'],
                                      pi_vector=all_named_args['pi_vector'])
    if 'newman' in which_methods:
        out_data = scores_storage.create_and_store_array('newman')
        compute_scores_with_transform(pairs_generator, adj_matrix,
                                      transforms_for_dot_prods.newman_transform, out_data,
                                      print_timing=print_timing, num_docs=all_named_args['num_docs'],
                                      pi_vector=all_named_args['pi_vector'])
    if 'shared_weight11' in which_methods:
        out_data = scores_storage.create_and_store_array('shared_weight11')
        compute_scores_with_transform(pairs_generator, adj_matrix,
                                      transforms_for_dot_prods.shared_weight11_transform, out_data,
                                      print_timing=print_timing,
                                      pi_vector=all_named_args['pi_vector'])

    if 'weighted_corr' in which_methods:
        out_data = scores_storage.create_and_store_array('weighted_corr')
        if sparse.isspmatrix(adj_matrix):
            # keep it sparse; can't use fastest method
            compute_scores_from_terms(pairs_generator, adj_matrix, wc_terms, out_data,
                                      pi_vector=all_named_args['pi_vector'],
                                      num_affils=adj_matrix.shape[1], print_timing=print_timing)

        else:
            compute_scores_with_transform(pairs_generator, adj_matrix,
                                          transforms_for_dot_prods.wc_transform, out_data,
                                          print_timing=print_timing,
                                          pi_vector=all_named_args['pi_vector'])

        if run_all_implementations:
            extra_implementations.simple_weighted_corr_sparse(pairs_generator, adj_matrix, out_data,
                                                              pi_vector=all_named_args['pi_vector'],
                                                              print_timing=print_timing)
            if not sparse.isspmatrix(adj_matrix):
                extra_implementations.simple_only_weighted_corr(pairs_generator, adj_matrix, out_data,
                                                                pi_vector=all_named_args['pi_vector'],
                                                                print_timing=print_timing)

    if 'weighted_corr_exp' in which_methods:
        out_data = scores_storage.create_and_store_array('weighted_corr_exp')
        if sparse.isspmatrix(adj_matrix):
            # keep it sparse; can't use fastest method
            simple_only_wc_exp_scores(pairs_generator, adj_matrix, all_named_args['exp_model'], out_data,
                                      print_timing=print_timing)

        else:
            compute_scores_with_transform(pairs_generator, adj_matrix,
                                          transforms_for_dot_prods.wc_exp_transform, out_data,
                                          exp_model=all_named_args['exp_model'],
                                          print_timing=print_timing)

    if 'shared_weight1100' in which_methods:
        out_data = scores_storage.create_and_store_array('shared_weight1100')
        compute_scores_from_terms(pairs_generator, adj_matrix, shared_weight1100_terms, out_data,
                                  pi_vector=all_named_args['pi_vector'], print_timing=print_timing)
    if 'mixed_pairs' in which_methods:
        for mp_sim in all_named_args['mixed_pairs_sims']:
            method_name = 'mixed_pairs_' + str(mp_sim)
            out_data = scores_storage.create_and_store_array(method_name)
            compute_scores_from_terms(pairs_generator, adj_matrix, mixed_pairs_terms, out_data,
                                      pi_vector=all_named_args['pi_vector'], sim=mp_sim,
                                      print_timing=print_timing)

    if 'hamming' in which_methods:
        back_compat = all_named_args.get('back_compat', False)
        out_data = scores_storage.create_and_store_array('hamming', dtype = int if back_compat else float)
        scoring_methods_fast.hamming_from_sharedsize(pairs_generator, adj_matrix, scores_storage, out_data,
                                print_timing=print_timing, back_compat=back_compat)
    if 'jaccard' in which_methods:
        out_data = scores_storage.create_and_store_array('jaccard')
        scoring_methods_fast.jaccard_from_sharedsize(pairs_generator, adj_matrix, scores_storage, out_data,
                                print_timing=print_timing, back_compat=back_compat)
    if 'pearson' in which_methods:
        out_data = scores_storage.create_and_store_array('pearson')
        scoring_methods_fast.simple_only_pearson(pairs_generator, adj_matrix, out_data, print_timing=print_timing)



    if run_all_implementations:
        extra_implementations.run_extra_implementations2(pairs_generator, adj_matrix, which_methods, scores_storage,
                                                         print_timing=print_timing, **all_named_args)

    start = timer()
    scores_storage.to_csv_gz(outfile_csv_gz, pairs_generator, adj_matrix)
    end = timer()
    if print_timing:
        print("saved scores to outfile " + outfile_csv_gz + ": " + str(end - start) + " secs")



def item_ids(pairs_generator):
    item1, item2 = [], []
    for (_, _, item1_id, item2_id, _, _) in pairs_generator:
        item1.append(item1_id)
        item2.append(item2_id)
    return(item1, item2)


def separate_faiss_methods(which_methods, faiss_preferred, is_sparse):
    if is_sparse:
        print("Can't use FAISS, because adjacency matrix was provided as sparse")
        return which_methods, []

    methods_for_faiss = set(x for x in which_methods if x[-6:] == '_faiss')  # honor the label
    try:
        import scoring_with_faiss
        faiss_avail = True
    except ImportError:
        faiss_avail = False
        print("FAISS not installed")
        if len(methods_for_faiss):
            print("skipping methods " + str(methods_for_faiss))

    our_methods_in_faiss = methods_for_faiss.copy()  # ours: what to remove from our list.
    if faiss_preferred and faiss_avail:
        # note: my cosine method for faiss isn't scaling as well as sklearn's. Only use it if explicitly asked.
        faiss_preferred_methods = set(scoring_with_faiss.all_faiss_methods) - set(('cosine_faiss', 'cosineIDF_faiss'))
        for method in which_methods:
            if method + '_faiss' in faiss_preferred_methods:
                methods_for_faiss.add(method + '_faiss')
                our_methods_in_faiss.add(method)
    if len(our_methods_in_faiss) > 0:
        print("Using FAISS for: " + str(list(our_methods_in_faiss)))
    which_methods = set(which_methods) - our_methods_in_faiss
    return which_methods, methods_for_faiss


# Turns out the scoring functions can be ~10x faster when the function is written as: transform the adjacency matrix, then
# take dot products of the row pairs.
def compute_scores_with_transform(pairs_generator, adj_matrix, transf_func, scores_out, print_timing=False, **named_args_to_func):
    start = timer()
    transformed_mat = transf_func(adj_matrix, **named_args_to_func)
    # scores = []
    for (i, j, _, _, pair_x, pair_y) in pairs_generator(transformed_mat):
        scores_out[i,j] = (pair_x.dot(pair_y))
    end = timer()
    if print_timing:
        print(transf_func.__name__ + ": " + str(end - start) + " secs")
    # return scores


# Basic formula, no tricks (yet)  # todo: test
def simple_only_wc_exp_scores(pairs_generator, adj_matrix, exp_model, scores_out, print_timing=False):
    start = timer()
    for (i, j, _, _, pair_x, pair_y) in pairs_generator(adj_matrix):
        p_rowx = exp_model.edge_prob_row(i)
        p_rowy = exp_model.edge_prob_row(j)
        score = np.sum( (pair_x - p_rowx) * (pair_y - p_rowy)/ np.sqrt(p_rowx * p_rowy * (1-p_rowx) * (1-p_rowy))) / len(pair_x)
        scores_out[i,j] = (score)

    end = timer()
    if print_timing:
        print("simple_only_wc_exp_scores: " + str(end - start) + " secs")


# This version: rather than explicitly computing the scores for 1/0 terms, have a base_score that assume all entries
# are 1/0s, and adjust it for 11 and 00s.
def compute_scores_from_terms(pairs_generator, adj_matrix, scores_bi_func, scores_out, print_timing=False, **named_args_to_func):
    if not sparse.isspmatrix(adj_matrix):
        return compute_scores_from_terms_dense(pairs_generator, adj_matrix, scores_bi_func, scores_out,
                                               print_timing=print_timing, **named_args_to_func)

    start = timer()
    terms_for_11, value_10, terms_for_00 = scores_bi_func(**named_args_to_func)
    base_score = value_10 * adj_matrix.shape[1]
    terms_for_11 -= value_10
    terms_for_00 -= value_10
    for (i, j, _, _, pair_x, pair_y) in pairs_generator(adj_matrix):
        sum_11 = terms_for_11.dot(pair_x * pair_y)
        sum_00 = terms_for_00.dot(np.logical_not(pair_x) * np.logical_not(pair_y))
        scores_out[i,j] = (base_score + sum_11 + sum_00 )

    end = timer()
    if print_timing:
        print(scores_bi_func.__name__ + ": " + str(end - start) + " secs")


# quick attempt to do like compute_faiss_terms_scores() -- only possible for dense
def compute_scores_from_terms_dense(pairs_generator, adj_matrix, scores_bi_func, scores_out, print_timing=False, **named_args_to_func):
    start = timer()
    terms_for_11, value_10, terms_for_00 = scores_bi_func(**named_args_to_func)
    base_score = value_10 * adj_matrix.shape[1]
    terms_for_11 -= value_10
    terms_for_00 -= value_10

    adj1 = adj_matrix * np.sqrt(terms_for_11)
    adj2 = np.logical_not(adj_matrix) * np.sqrt(terms_for_00)

    for (i, j, _, _, pair_x, pair_y) in pairs_generator(adj1):
        sum_11 = pair_x.dot(pair_y)
        sum_00 = adj2[i,].dot(adj2[j,])
        scores_out[i,j] = (base_score + sum_11 + sum_00)

    end = timer()
    if print_timing:
        print("dense " + scores_bi_func.__name__ + ": " + str(end - start) + " secs")


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




# Current status on speed (notes to self):
# -Dense matrix calcs are much faster. It's a time vs. space tradeoff.
# -Slowest methods, when using sparse adj_mat: anything that doesn't use *transform or sklearn.
#   shared_weight1100, mixedPairs & weightedCorr; pearson
# -Slowest methods when using dense adj_mat:
#   pearson
# (new: Hamming & Jaccard now just add a smidgen of time beyond shared_size.)
# -when calling score_pairs with run_all_implementations=False, it uses the one that's been fastest on the example data,
#  but best one could change on different sized data sets, and depending on whether matrix is sparse or dense.

# Ways to improve:
# -sklearn's cosine is far faster than my other methods. Could use sklearn's linear_product for dot products (etc) --
#  likely also faster than my code. It preserves sparse matrices.
# But: sklearn methods do all_pairs by default. (Scoring pairs as they come out of the pairs_generator, as my code
# does, is probably terrible for speed.) Scoring only a subset would be non-trivial.
#
# Not fussing with these right now, because faiss is yet faster. (Betting on being able to use it even with
# sparse matrices.)

# Note: it's seeming like dense + memmap is going to be preferred over sparse matrix methods. I.e., it's worth using disk
# space to save time. Do I want to drop support for sparse matrices?