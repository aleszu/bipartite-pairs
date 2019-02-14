import faiss
import numpy as np
import pandas
from scipy import sparse
from timeit import default_timer as timer
import transforms_for_dot_prods

# note: for now, keep these names disjoint from the normal ones in scoring_methods -- that's how we tell the difference
all_faiss_methods = ['shared_size_faiss', 'adamic_adar_faiss', 'newman_faiss', 'shared_weight11_faiss',
                       'weighted_corr_faiss', 'weighted_corr_exp_faiss']
# todo? add cosine*, pearson

# note: faiss methods all require dense version of adj matrix

# wrapper for all-pairs exact distances using "Flat" index
# Returns a DataFrame with of all pairs
def score_pairs_faiss_all_exact(adj_matrix, which_methods, print_timing=False, **all_named_args):
    return score_pairs_faiss(adj_matrix, which_methods, how_many_neighbors=-1, print_timing=print_timing, **all_named_args)

# This function can be called in two modes:
# (1) all pairs: each method returns a distance matrix, converted into a DataFrame at the end
# (2) k-nearest neighbors: each method returns a set of distances, at the end converted into a DataFrame of all pairs
# any method returned
def score_pairs_faiss(adj_matrix, which_methods, how_many_neighbors=-1, print_timing=False, **all_named_args):
    scores = {}     # dictionary of sparse matrices
    if which_methods == 'all':
        which_methods = all_faiss_methods

    # each method returns either a distance matrix (if all neighbors) or a sparse distance matrix (if just k-nearest)

    if 'shared_size_faiss' in which_methods:
        scores['shared_size_faiss'] = compute_faiss_dotprod_distances(adj_matrix,
                                                              transforms_for_dot_prods.shared_size_transform,
                                                              how_many_neighbors, print_timing=print_timing,
                                                              back_compat=all_named_args.get('back_compat', False))
    if 'adamic_adar_faiss' in which_methods:
        scores['adamic_adar_faiss'] = compute_faiss_dotprod_distances(adj_matrix,
                                                                transforms_for_dot_prods.adamic_adar_transform,
                                                                how_many_neighbors, print_timing = print_timing,
                                                                num_docs = all_named_args['num_docs'],
                                                                pi_vector = all_named_args['pi_vector'])
    if 'newman_faiss' in which_methods:
        scores['newman_faiss'] = compute_faiss_dotprod_distances(adj_matrix,
                                                            transforms_for_dot_prods.newman_transform,
                                                            how_many_neighbors, print_timing=print_timing, num_docs=all_named_args['num_docs'],
                                                            pi_vector=all_named_args['pi_vector'])
    if 'shared_weight11_faiss' in which_methods:
        scores['shared_weight11_faiss'] = compute_faiss_dotprod_distances(adj_matrix,
                                                                  transforms_for_dot_prods.shared_weight11_transform,
                                                                  how_many_neighbors, print_timing=print_timing,
                                                                  pi_vector=all_named_args['pi_vector'])
    if 'weighted_corr_faiss' in which_methods:
        scores['weighted_corr_faiss'] = compute_faiss_dotprod_distances(adj_matrix, transforms_for_dot_prods.wc_transform,
                                                                        print_timing=print_timing,
                                                                        how_many_neighbors=how_many_neighbors,
                                                                        pi_vector=all_named_args['pi_vector'])
    if 'weighted_corr_exp_faiss' in which_methods:
        scores['weighted_corr_exp_faiss'] = compute_faiss_dotprod_distances(adj_matrix, transforms_for_dot_prods.wc_exp_transform,
                                                                            how_many_neighbors, print_timing=print_timing,
                                                                            exp_model = all_named_args['exp_model'])

    if len(scores):
        if how_many_neighbors == -1:
            # iterate through all pairs --> array
            scores_array = np.empty((adj_matrix.shape[0] * (adj_matrix.shape[0]-1)/2, 2 + len(scores)))
            cnt = 0
            for i in range(adj_matrix.shape[0]):
                for j in range(i+1, adj_matrix.shape[0]):
                    scores_array[cnt] = [i, j] + [res_mat[i,j] for res_mat in scores.values()]
                    cnt += 1

            return pandas.DataFrame(scores_array, columns = ['item1', 'item2'] + scores.keys())

        else:
            # convert distance matrices to a single data frame with pairs + scores as columns (some scores will be blank)
            # (note: making dok_mat was probably unnecessary b/c just using it as a dict)
            scores_frame = pandas.DataFrame( {method:pandas.Series(dict(dok_mat.items()))
                                              for (method, dok_mat) in scores.items()} )
            return scores_frame.rename_axis(index=['item1', 'item2']).reset_index()
    else:
        return None


def convert_dist_results_to_matrix(distances, neighbors):
    """Converts nearest-neighbor results to a matrix of pairwise distances.
    Using dictionary-of-keys approach because neighbors may include (i,j) but not (j,i); we'll take either.

    :param distances: matrix returned by index.search()
    :param neighbors: matrix returned by index.search()
    :return: a sparse.dok_matrix of pairwise distances
    """
    mtx = sparse.dok_matrix((distances.shape[0], distances.shape[0]), dtype=np.float32)
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):  # jth closest neighbor for item i
            neighb = neighbors[i,j]
            # insert (i,neighb) such that i < neighb
            min_in = min(i, neighb)
            max_in = max(i, neighb)
            if i != neighb and mtx[min_in, max_in] == 0:
                mtx[min_in, max_in] = distances[i,j]
    return mtx


# For faiss, transf_func needs to return a dense matrix, so:
# --> this function needs 'make_dense=True' to be passed in (for the *_transform functions that allow it).
def compute_faiss_dotprod_distances(adj_matrix, transf_func, how_many_neighbors, print_timing=False, **named_args_to_func):
    start0 = timer()
    # transformation
    start = timer()
    transformed_mat = np.ascontiguousarray(transf_func(adj_matrix, **named_args_to_func).astype('float32'))
    if print_timing:
        end = timer()
        print "\t" + transf_func.__name__ + " of adj_matrix: " + str(end - start) + " secs"

    # index construction
    start = timer()
    faiss_index = faiss.IndexFlatIP(transformed_mat.shape[1])
    faiss_index.add(transformed_mat)
    if print_timing and False:
        end = timer()
        print "\t" + "faiss index construction: " + str(end - start) + " secs"

    start = timer()
    # querying for neighbors
    if how_many_neighbors == -1:
        how_many_neighbors = transformed_mat.shape[0]  # all

    if how_many_neighbors > 0:
        distances, neighbors = faiss_index.search(transformed_mat, how_many_neighbors)
        if print_timing and False:
            end = timer()
            print "\t" + "faiss querying for " + transf_func.__name__ + ": " + str(end - start) + " secs"

        dists_matrix = convert_dist_results_to_matrix(distances, neighbors)

    else:  # get all distances immediately from faiss
        labels = np.array([range(transformed_mat.shape[0]) for i in range(transformed_mat.shape[0])]) # n rows of 0..n
        dists_matrix = np.empty((transformed_mat.shape[0], transformed_mat.shape[0]), dtype='float32')
        faiss_index.compute_distance_subset(transformed_mat.shape[0], faiss.swig_ptr(transformed_mat), transformed_mat.shape[0],
                                      faiss.swig_ptr(dists_matrix), faiss.swig_ptr(labels))

    if print_timing:
        end = timer()
        print "total compute_faiss_dotprod_distances for " + transf_func.__name__ + ": " + str(end - start0) + " secs"

    return dists_matrix
