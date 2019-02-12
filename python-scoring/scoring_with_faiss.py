import faiss
import sys
import numpy as np
import pandas
from scipy import sparse
from timeit import default_timer as timer

sys.path.append("../python-scoring")  # (see tests.py for syntax)
                                      # pycharm-friendly call is below. (Need sys.path call to avoid errors
                                      # from imports within score_data.)
import imp
scoring_methods = imp.load_source("scoring_methods", "../python-scoring/scoring_methods.py")

# note: for now, keep these names disjoint from the normal ones in scoring_methods -- that's how we tell the difference
all_faiss_methods = ['cosine_faiss', 'cosineIDF_faiss', 'shared_size_faiss',
                       'weighted_corr_faiss', 'weighted_corr_exp_faiss'] # todo: others too

# note: faiss methods all make adj matrix dense


def score_pairs_faiss(adj_matrix, which_methods, how_many_neighbors=-1, print_timing=False, **all_named_args):
    scores = {}     # dictionary of sparse matrices
    if which_methods == 'all':
        which_methods = all_faiss_methods

    # each method: run, get back a sparse distance matrix
    # if 'cosine_faiss' in which_methods:
    #     scores['cosine_faiss'] = compute_faiss_distances(adj_matrix, cosine_tranform, print_timing=print_timing)

    if 'weighted_corr_faiss' in which_methods:
        scores['weighted_corr_faiss'] = compute_faiss_dotprod_distances(adj_matrix, scoring_methods.wc_transform,
                                                                        print_timing=print_timing,
                                                                        how_many_neighbors=how_many_neighbors,
                                                                        pi_vector=all_named_args['pi_vector'])
    # if 'weighted_corr_exp_faiss' in which_methods:
    #     scores['weighted_corr_exp_faiss'] = compute_faiss_distances(adj_matrix, wc_exp_transform,
    #                                                             print_timing=print_timing,
    #                                                             exp_model = all_named_args['exp_model'])

    if len(scores):
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
            min_in = min(i, neighb)
            max_in = max(i, neighb)
            if i != neighb and mtx[min_in, max_in] == 0:
                # insert (i,neighb) such that i < neighb
                mtx[min_in, max_in] = distances[i,j]
    return mtx


# For faiss, transf_func needs to return a dense matrix, so:
# --> this function needs 'make_dense=True' to be passed in (for the *_transform functions that allow it).
def compute_faiss_dotprod_distances(adj_matrix, transf_func, how_many_neighbors, print_timing=False, **named_args_to_func):
    start0 = timer()
    # transformation
    start = timer()
    transformed_mat = transf_func(adj_matrix, **named_args_to_func).astype('float32')
    end = timer()
    if print_timing:
        print transf_func.__name__ + ": " + str(end - start) + " secs"

    # index construction
    start = timer()
    faiss_index = faiss.IndexFlatIP(transformed_mat.shape[1])
    faiss_index.add(np.ascontiguousarray(transformed_mat))
    end = timer()
    if print_timing:
        print "faiss index construction for " + transf_func.__name__ + ": " + str(end - start) + " secs"

    # querying for neighbors
    start = timer()
    if how_many_neighbors == -1:
        how_many_neighbors = transformed_mat.shape[0]   # all
    distances, neighbors = faiss_index.search(np.ascontiguousarray(transformed_mat), how_many_neighbors)
    end = timer()
    if print_timing:
        print "faiss querying for " + transf_func.__name__ + ": " + str(end - start) + " secs"

    dists_matrix = convert_dist_results_to_matrix(distances, neighbors)
    end = timer()
    if print_timing:
        print "total compute_faiss_dotprod_distances for " + transf_func.__name__ + ": " + str(end - start0) + " secs"

    return dists_matrix
