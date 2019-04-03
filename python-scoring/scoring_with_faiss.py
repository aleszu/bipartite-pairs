from __future__ import print_function
from builtins import str, range
import faiss
import numpy as np
from scipy import sparse
from timeit import default_timer as timer
import tempfile
import transforms_for_dot_prods
import scoring_methods
import magic_dictionary

# note: for now, keep these names disjoint from the normal ones in scoring_methods -- that's how we tell the difference
all_faiss_methods = ['shared_size_faiss', 'adamic_adar_faiss', 'newman_faiss', 'shared_weight11_faiss',
                       'weighted_corr_faiss', 'weighted_corr_exp_faiss', 'mixed_pairs_faiss', 'pearson_faiss',
                     'cosine_faiss', 'cosineIDF_faiss', 'shared_weight1100_faiss']

# Note: faiss methods all require dense version of adj matrix. (Converting it to work on sparse (in batches) will
# require revamping compute_faiss_terms_scores(), too.)
# To keep things simple, I'll ask the calling function, score_pairs, to only send in dense matrices. Meaning this
# package will generally only be called from run_and_eval(make_dense=True).

# wrapper for all-pairs exact distances using "Flat" index
# Returns a DataFrame with of all pairs
def score_pairs_faiss_all_exact(adj_matrix, which_methods, scores_storage_magic_d, print_timing=False, **all_named_args):
    return score_pairs_faiss(adj_matrix, which_methods, scores_storage_magic_d, how_many_neighbors=-1,
                             print_timing=print_timing, **all_named_args)

# This function can be called in two modes:
# (1) all pairs: each method returns a distance matrix, converted into a DataFrame at the end
# (2) k-nearest neighbors: each method returns a set of distances, at the end converted into a DataFrame of all pairs
# any method returned
def score_pairs_faiss(adj_matrix, which_methods, scores_storage, how_many_neighbors=-1, print_timing=False, **all_named_args):

    if which_methods == 'all':
        which_methods = all_faiss_methods

    # if we're using on-disk memory instead of RAM, do the same for intermediate variables
    use_memmap = isinstance(scores_storage, magic_dictionary.onDiskDict)

    # each method returns either a distance matrix (if all neighbors) or a sparse distance matrix (if just k-nearest)

    if 'shared_size_faiss' in which_methods:
        out_data = scores_storage.create_and_store_array('shared_size_faiss', dtype=np.float32)
        compute_faiss_dotprod_distances(adj_matrix, transforms_for_dot_prods.shared_size_transform,
                                        how_many_neighbors, out_data, use_memmap=use_memmap, print_timing=print_timing,
                                        back_compat=all_named_args.get('back_compat', False))
    if 'pearson_faiss' in which_methods:
        out_data = scores_storage.create_and_store_array('pearson_faiss', dtype=np.float32)
        compute_faiss_dotprod_distances(adj_matrix, transforms_for_dot_prods.pearson_transform,
                                        how_many_neighbors, out_data, use_memmap=use_memmap, print_timing=print_timing)
    if 'cosine_faiss' in which_methods:
        out_data = scores_storage.create_and_store_array('cosine_faiss', dtype=np.float32)
        compute_faiss_dotprod_distances(adj_matrix, transforms_for_dot_prods.cosine_transform,
                                        how_many_neighbors, out_data, use_memmap=use_memmap, print_timing=print_timing)
    if 'cosineIDF_faiss' in which_methods:
        idf_weights = np.log(1 / all_named_args['pi_vector'])
        out_data = scores_storage.create_and_store_array('cosineIDF_faiss', dtype=np.float32)
        compute_faiss_dotprod_distances(adj_matrix, transforms_for_dot_prods.cosine_weights_transform,
                                        how_many_neighbors, out_data, use_memmap=use_memmap,
                                        weights=idf_weights, print_timing=print_timing)
    if 'adamic_adar_faiss' in which_methods:
        out_data = scores_storage.create_and_store_array('adamic_adar_faiss', dtype=np.float32)
        compute_faiss_dotprod_distances(adj_matrix, transforms_for_dot_prods.adamic_adar_transform,
                                        how_many_neighbors, out_data, use_memmap=use_memmap, print_timing=print_timing,
                                        num_docs=all_named_args['num_docs'], pi_vector=all_named_args['pi_vector'])
    if 'newman_faiss' in which_methods:
        out_data = scores_storage.create_and_store_array('newman_faiss', dtype=np.float32)
        compute_faiss_dotprod_distances(adj_matrix, transforms_for_dot_prods.newman_transform,
                                        how_many_neighbors, out_data, use_memmap=use_memmap, print_timing=print_timing,
                                        num_docs=all_named_args['num_docs'], pi_vector=all_named_args['pi_vector'])
    if 'shared_weight11_faiss' in which_methods:
        out_data = scores_storage.create_and_store_array('shared_weight11_faiss', dtype=np.float32)
        compute_faiss_dotprod_distances(adj_matrix, transforms_for_dot_prods.shared_weight11_transform,
                                        how_many_neighbors, out_data, use_memmap=use_memmap, print_timing=print_timing,
                                        pi_vector=all_named_args['pi_vector'])
    if 'weighted_corr_faiss' in which_methods:
        out_data = scores_storage.create_and_store_array('weighted_corr_faiss', dtype=np.float32)
        compute_faiss_dotprod_distances(adj_matrix, transforms_for_dot_prods.wc_transform,
                                        how_many_neighbors, out_data, use_memmap=use_memmap, print_timing=print_timing,
                                        pi_vector=all_named_args['pi_vector'])
    if 'weighted_corr_exp_faiss' in which_methods:
        out_data = scores_storage.create_and_store_array('weighted_corr_exp_faiss', dtype=np.float32)
        compute_faiss_dotprod_distances(adj_matrix, transforms_for_dot_prods.wc_exp_transform,
                                        how_many_neighbors, out_data, use_memmap=use_memmap, print_timing=print_timing,
                                        exp_model = all_named_args['exp_model'])

    # these require computing all pairs
    if 'shared_weight1100_faiss' in which_methods and how_many_neighbors == -1:
        out_data = scores_storage.create_and_store_array('shared_weight1100_faiss', dtype=np.float32)
        compute_faiss_terms_scores(adj_matrix, scoring_methods.shared_weight1100_terms, out_data,
                                   print_timing=print_timing, pi_vector=all_named_args['pi_vector'])
    if 'mixed_pairs_faiss' in which_methods and how_many_neighbors == -1:
        for mp_sim in all_named_args['mixed_pairs_sims']:
            method_name = 'mixed_pairs_faiss_' + str(mp_sim)
            out_data = scores_storage.create_and_store_array(method_name, dtype=np.float32)
            compute_faiss_terms_scores(adj_matrix, scoring_methods.mixed_pairs_terms, out_data,
                                       pi_vector=all_named_args['pi_vector'], sim=mp_sim, print_timing=print_timing)




def convert_dist_results_to_matrix(distances, neighbors, dists_matrix_out):
    """Converts nearest-neighbor results to a matrix of pairwise distances.
    Using dictionary-of-keys approach because neighbors may include (i,j) but not (j,i); we'll take either.

    :param distances: matrix returned by index.search()
    :param neighbors: matrix returned by index.search()
    :return: a sparse.dok_matrix of pairwise distances
    """
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):  # jth closest neighbor for item i
            neighb = neighbors[i,j]
            # insert (i,neighb) such that i < neighb
            min_in = min(i, neighb)
            max_in = max(i, neighb)
            if i != neighb and dists_matrix_out[min_in, max_in] == 0:
                dists_matrix_out[min_in, max_in] = distances[i,j]


def compute_faiss_dotprod_distances(adj_matrix, transf_func, how_many_neighbors, dists_matrix_out,
                                    use_memmap=False, print_timing=False, **named_args_to_func):
    start0 = timer()

    # transformation
    start = timer()
    if use_memmap:
        transformed_mat = np.memmap(tempfile.NamedTemporaryFile(), dtype=np.float32, mode="w+", shape=adj_matrix.shape)
    else:
        transformed_mat = np.empty(adj_matrix.shape, dtype=np.float32)

    # Send in and get back a dense adj_matrix, regardless of arg "make_dense". A non-sparse array is by default
    # [C-]contiguous, so no more need to convert.
    transformed_mat[:] = transf_func(adj_matrix.toarray() if sparse.isspmatrix(adj_matrix) else adj_matrix,
                                     **named_args_to_func)[:]
    if print_timing:
        end = timer()
        print("\t" + transf_func.__name__ + " of adj_matrix: " + str(end - start) + " secs")

    # index construction
    start = timer()
    faiss_index = faiss.IndexFlatIP(transformed_mat.shape[1])
    faiss_index.add(transformed_mat)
    if print_timing and False:
        end = timer()
        print("\t" + "faiss index construction: " + str(end - start) + " secs")

    start = timer()
    # querying for neighbors
    if how_many_neighbors > 0:
        distances, neighbors = faiss_index.search(transformed_mat, how_many_neighbors)
        if print_timing and False:
            end = timer()
            print("\t" + "faiss querying for " + transf_func.__name__ + ": " + str(end - start) + " secs")

        convert_dist_results_to_matrix(distances, neighbors, dists_matrix_out)

    else:  # get all distances immediately from faiss
        tot_nodes = transformed_mat.shape[0]
        if use_memmap:
            labels = np.memmap(tempfile.NamedTemporaryFile(), dtype=int, mode="w+",
                                        shape=(adj_matrix.shape[0],adj_matrix.shape[0]))
        else:
            labels = np.empty((adj_matrix.shape[0],adj_matrix.shape[0]), dtype=int)

        labels[:] = np.tile(range(tot_nodes), (tot_nodes, 1))  # n rows of 0..n
        faiss_index.compute_distance_subset(tot_nodes, faiss.swig_ptr(transformed_mat), tot_nodes,
                                      faiss.swig_ptr(dists_matrix_out), faiss.swig_ptr(labels))

    if print_timing:
        end = timer()
        print("total compute_faiss_dotprod_distances for " + transf_func.__name__[:-10] + ": " + str(end - start0) + " secs")



# For functions that have different terms for 1/1, 1/0, and 0/0 components.
# Only applies when we're getting all neighbors.
def compute_faiss_terms_scores(adj_matrix, scores_bi_func, dists_matrix_out, use_memmap=False,
                               print_timing=False, **named_args_to_func):
    start = timer()
    # terms_* vars are vectors (for .dot(row)), value_10 is a scalar
    terms_for_11, value_10, terms_for_00 = scores_bi_func(**named_args_to_func)

    base_score = value_10 * adj_matrix.shape[1]  # start with score(1/0)
    terms_for_11 -= value_10   # for 1/1, add score(1/1) - score(1/0)
    terms_for_00 -= value_10

    if use_memmap:
        adj1 = np.memmap(tempfile.NamedTemporaryFile(), dtype=float, mode="w+", shape=adj_matrix.shape)
        dists1 = np.memmap(tempfile.NamedTemporaryFile(), dtype=np.float32, mode="w+",
                           shape=(adj_matrix.shape[0], adj_matrix.shape[0]))
        dists2 = np.memmap(tempfile.NamedTemporaryFile(), dtype=np.float32, mode="w+",
                           shape=(adj_matrix.shape[0], adj_matrix.shape[0]))
    else:
        adj1 = np.empty(adj_matrix.shape)
        dists1 = np.empty((adj_matrix.shape[0], adj_matrix.shape[0]), dtype=np.float32)
        dists2 = np.empty((adj_matrix.shape[0], adj_matrix.shape[0]), dtype=np.float32)

    # old way:
    #   sum_11 = terms_for_11.dot(pair_x * pair_y)
    #   sum_00 = terms_for_00.dot(np.logical_not(pair_x) * np.logical_not(pair_y))
    #   scores.append(base_score + sum_11 + sum_00)
    # rewritten as the sum of two dot products:

    adj1[:] = (adj_matrix * np.sqrt(terms_for_11))[:]
    # identity_func = lambda x: x
    compute_faiss_dotprod_distances(adj1, lambda x: x, how_many_neighbors=-1, dists_matrix_out=dists1, print_timing=False)

    adj1[:] = (np.logical_not(adj_matrix) * np.sqrt(terms_for_00))[:]
    compute_faiss_dotprod_distances(adj1, lambda x: x, how_many_neighbors=-1, dists_matrix_out=dists2, print_timing=False)

    dists_matrix_out[:] = (dists1 + dists2 + base_score)[:]     # copy values
    if print_timing:
        end = timer()
        print("total compute_faiss_terms_scores for " + scores_bi_func.__name__ + ": " + str(end - start) + " secs")


