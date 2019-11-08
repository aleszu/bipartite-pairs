from __future__ import print_function

import os
import tempfile

from builtins import map, str, range
from scipy.io import mmread
from scipy import sparse
import gzip
from sklearn.metrics import roc_auc_score
import sys
import numpy as np
import pandas as pd
from functools import partial

import scoring_methods
from bipartite_fitting import learn_graph_models

# This file: If run from the command line, script simply reads data file, saves pairs with scores.
# General purpose functions: score_data(), get_item_likelihoods, and write_item_likelihoods().
# For expts with labeled data, use run_and_eval(): you provide true pairs, it computes AUCs, and it has options for
# loading and manipulating phi.


# Iterates through all pairs of matrix rows, in the form (i, j) where i < j
# Rows output are always of type numpy.ndarray()
# Returns (row1_index, row2_index, row1_label (differs from row1_index if row_labels param sent), row2_label, row1, row2)
# (although the row "labels" this codebase creates are actually integers)
def gen_all_pairs(my_adj_mat, row_labels=None):
    num_rows = my_adj_mat.shape[0]
    is_sparse = sparse.isspmatrix(my_adj_mat)
    is_numpy_matrix = (type(my_adj_mat) == np.matrix)   # this type results from operations using sparse + ndarray
    for i in range(num_rows):
        if is_sparse:
            rowi = my_adj_mat.getrow(i).toarray()[0]  # toarray() gives 2-d ndarray, [0] to flatten
        else:   # already an ndarray(), possibly a matrix()
            rowi = my_adj_mat[i,]
            if is_numpy_matrix:
                rowi = rowi.A1

        for j in range(i + 1, num_rows):
            if is_sparse:
                rowj = my_adj_mat.getrow(j).toarray()[0]
            else:
                rowj = my_adj_mat[j,]
                if is_numpy_matrix:
                    rowj = rowj.A1
            if row_labels is None:
                yield (i, j, i, j, rowi, rowj)
            else:
                yield (i, j, row_labels[i], row_labels[j], rowi, rowj)


# For when we're just looping over indices, don't need to access rows
# note: this returns a generator function. in contrast, gen_all_pairs is a generator function and returns a generator object.
# syntax: "for x in <generator object>:" vs. "for x in <generator function>(<possible args>):"
def ij_gen(max, row_labels=None):
    if row_labels is not None:
        return ((i, j, row_labels[i], row_labels[j], 0, 0) for i in range(max) for j in range((i + 1), max))
    else:
        return((i, j, i, j, 0, 0) for i in range(max) for j in range((i+1),max))


# Infile must be in "matrix market" format and gzipped
# Returns a "compressed sparse column" matrix (for efficient column slicing)
def load_adj_mat(datafile_mmgz, make_binary = True):
    with gzip.open(datafile_mmgz, 'r') as fp_mm:
        adj_mat = mmread(fp_mm).astype(int, copy=False)  # creates a scipy.sparse.coo_matrix, if matrix was sparse
    if make_binary:     # change edge weights to 1
        adj_mat = adj_mat.astype(bool, copy=False).astype('int8', copy=False)

    print("Read data: " + str(adj_mat.shape[0]) + " items, " + str(adj_mat.shape[1]) + " affiliations")
    return adj_mat.tocsc()


# Returns a numpy.ndarray with 1 axis (.ndim = 1)
def learn_pi_vector(adj_mat):
    pi_orig = adj_mat.sum(axis=0) / float(adj_mat.shape[0])  # a numpy matrix, a type that always stays 2D unless we call .tolist()[0] or np.asarray()
    return np.asarray(pi_orig).squeeze()


def remove_boundary_nodes(adj_mat, pi_vector = None, flip_high_ps=False, orig_row_labels=None,
                          remove_boundary_items=True, remove_boundary_affils=True):
    """
    Replaces adjust_pi_vector. This version removes both items and affils that are all 0's or all 1's,
    recursively (until there are no more). Returns updated adj_mat and also the "row labels" (i.e.,
    original indices) of the items that remain. We can compute the pi_vector anytime from the adj_mat, but keeping
    it as an optional argument in case it was loaded from a file rather than learned. (Note: don't load pi_vector
    from a file when using remove_boundary_items=True; behavior may not be as expected.)
    :param pi_vector:
    :param adj_mat:
    :param pi_vector:
    :param flip_high_ps:
    :return:
    """
    adj_mat_mod = adj_mat
    if orig_row_labels is None:
        row_labels = np.arange(adj_mat.shape[0])
    else:
        row_labels = orig_row_labels

    if pi_vector is not None:
        pi_vector_mod = pi_vector.copy()
    else:
        pi_vector_mod = learn_pi_vector(adj_mat)
            # or in one line: np.asarray(adj_mat.sum(axis=0)).squeeze() / float(adj_mat.shape[0])

    need_to_check = True
    # initialize
    items_to_keep = np.full(adj_mat.shape[0], True)
    affils_to_keep = np.full(adj_mat.shape[1], True)
    while (need_to_check):
        need_to_check = False   # only one condition turns it back on

        if remove_boundary_affils:
            # Remove affils we don't want
            affil_degrees = np.asarray(adj_mat_mod.sum(axis=0)).squeeze()
            affils_to_keep = np.logical_and(affil_degrees > 0, affil_degrees < adj_mat_mod.shape[0])
            if affils_to_keep.sum() < len(affil_degrees):
                which_nonzero = np.nonzero(affils_to_keep)      # returns a tuple (immutable list) holding 1 element: an ndarray of indices
                adj_mat_mod = adj_mat_mod[:, which_nonzero[0]]
                pi_vector_mod = pi_vector_mod[which_nonzero]        # since pi_vector is also an ndarray, the slicing can take the tuple

        if remove_boundary_items:
            # Remove items we don't want
            item_degrees = np.asarray(adj_mat_mod.sum(axis=1)).squeeze()
            items_to_keep = np.logical_and(item_degrees > 0, item_degrees < adj_mat_mod.shape[1])
            if items_to_keep.sum() < len(item_degrees):
                which_nonzero = np.nonzero(items_to_keep)
                adj_mat_mod = adj_mat_mod[which_nonzero[0], :]
                row_labels = row_labels[which_nonzero]
                # update pi_vector, provided we're using one fit to this data
                if pi_vector is None:
                    pi_vector_mod = learn_pi_vector(adj_mat_mod)
                if remove_boundary_affils:
                    need_to_check = True

    print("Keeping " + ("all " if (items_to_keep.sum() == adj_mat.shape[0]) else "") + str(items_to_keep.sum()) +
          " items and " + ("all " if (affils_to_keep.sum() == adj_mat.shape[1]) else "") \
          + str(affils_to_keep.sum()) + " affils")

    affil_degrees = np.asarray(adj_mat_mod.sum(axis=0)).squeeze()
    cmpts_to_flip = affil_degrees > .5 * adj_mat_mod.shape[0]
    if flip_high_ps:
        print("Flipping " + str(cmpts_to_flip.sum()) + " components that had p_i > .5")
        print("(ok to ignore warning message produced)")
        which_nonzero = np.nonzero(cmpts_to_flip)
        pi_vector_mod[which_nonzero] = 1 - pi_vector_mod[which_nonzero]
        adj_mat_mod[:, which_nonzero[0]] = np.ones(adj_mat_mod[:, which_nonzero[0]].shape, dtype=adj_mat_mod.dtype) \
                                           - adj_mat_mod[:, which_nonzero[0]]

    else:
        print("fyi: leaving in the " + str(cmpts_to_flip.sum()) + " components with p_i > .5")

    # For speed, make sure to leave row_labels as None if it started that way and num items hasn't changed
    if orig_row_labels is None and items_to_keep.sum() == adj_mat.shape[0]:
        row_labels = None

    return pi_vector_mod, adj_mat_mod.tocsr(), row_labels


def run_and_eval(adj_mat, true_labels_func, method_spec, evals_outfile,
                 pair_scores_outfile=None, mixed_pairs_sims='standard', add_exp_model=False,
                 make_dense=True, prefer_faiss=False, print_timing=False,
                 row_labels=None, pi_vector_to_use=None,
                 flip_high_ps=False, remove_boundary_items=True, remove_boundary_affils=True):
    """
    :param adj_mat:
    :param true_labels_func: identifies the true pairs, given a pairs_generator
    :param method_spec: list of method names OR the string 'all'
    :param evals_outfile:
    :param pair_scores_outfile:
    :param mixed_pairs_sims:
    :param add_exp_model:
    :param make_dense:
    :param prefer_faiss:
    :param print_timing:
    :param row_labels: used when adj_mat's indices differ from original row numbers
    :param pi_vector_to_use:
    :param flip_high_ps:
    :param remove_boundary_affils:
    :param remove_boundary_items:
    :return:
    """

    # note on sparse matrices: adj_mat is initially read in as "coo" format (coordinates of entries). Next few operations
    # will be by column, so it's returned from load_adj_mat as "csc" (compressed sparse column). Then, converted to
    # "csr" in adjust_pi_vector to make pair generation (row slicing) fast.

    pi_vector, adj_mat, row_labels = remove_boundary_nodes(adj_mat, pi_vector=pi_vector_to_use,
                                                           flip_high_ps=flip_high_ps, orig_row_labels=row_labels,
                                                           remove_boundary_items=remove_boundary_items,
                                                           remove_boundary_affils=remove_boundary_affils)
    if make_dense:
        adj_mat = adj_mat.toarray()

    # score pairs
    # (sending in all special args any methods might need)

    want_exp_model = add_exp_model or ('weighted_corr_exp' in method_spec) or \
                     ('weighted_corr_exp_faiss' in method_spec) or ('all' in method_spec)
    graph_models = learn_graph_models(adj_mat, bernoulli=True, pi_vector=pi_vector, exponential=want_exp_model,
                                       verbose = print_timing, max_iter_biment=50000)

    # In the future (anticipated for scaling up): first, run any methods that return a subset of pairs.
    # scores_subset =
    # Then make pairs_generator use the pairs in scores_subset.

    # Pairs generators. We need:
    # 1. Full version that accesses matrix rows, and cheap/efficient version that just gives row indices.
    # 2. To be able to call each of them multiple times (once per method).
    # 3. Full version must be able to take different adj_matrix arguments (transformed matrices).
    # 4. But row_labels arg should be wrapped into both, right here.

    # functools.partial lets us construct generators that are automatically reset w/orig args when called again.
    pairs_generator = partial(gen_all_pairs, row_labels=row_labels)         # this is a generator function. Call it with an arg to get generator object.
    pairs_gen_for_labels = partial(ij_gen, adj_mat.shape[0], row_labels)    # this too is a generator function. Call it w/o args to get generator object.
    # equivalent, but less elegant than functools.partial
    # def my_pairs_gen(adj_mat):
    #     return gen_all_pairs(adj_mat, row_labels)
    # pairs_generator = my_pairs_gen      # this is a generator function. Call it with an arg to get generator object.

    # outfile: even if caller didn't ask for one, we need a temporary one
    if pair_scores_outfile is None:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".csv.gz")
        initial_pairs_outfile = tf.name
        tf.close()
    else:
        initial_pairs_outfile = pair_scores_outfile

    scoring_methods.score_pairs(pairs_generator, adj_mat, method_spec,
                                outfile_csv_gz=initial_pairs_outfile, indices_gen=pairs_gen_for_labels,
                                pi_vector=graph_models['bernoulli'].affil_params,
                                exp_model=graph_models.get('exponential', None),
                                num_docs=adj_mat.shape[0], mixed_pairs_sims=mixed_pairs_sims,
                                print_timing=print_timing, prefer_faiss=prefer_faiss)
    # if scores_subset is not None:
    #     scores_data_frame = pd.merge(scores_subset, scores_data_frame, on=['item1', 'item2'])

    with gzip.open(initial_pairs_outfile, 'r') as fpin:
        scores_data_frame = pd.read_csv(fpin)

    method_names = set(scores_data_frame.columns.tolist()) - {'item1', 'item2'}
    scores_data_frame['label'] = list(map(int, true_labels_func(pairs_gen_for_labels())))

    # round pair scores at 15th decimal place so we don't get spurious diffs in AUCs when replicating
    scores_data_frame = scores_data_frame.round(decimals={method:15 for method in method_names})

    # save pair scores if desired
    if pair_scores_outfile is not None:
        scores_data_frame = scores_data_frame.reindex(columns=['item1', 'item2', 'label'] +
                                                              sorted(list(method_names - {'label'})), copy=False)
        scores_data_frame.to_csv(pair_scores_outfile, index=False, compression="gzip")
    else:
        os.remove(initial_pairs_outfile)

    # compute evals and save
    evals = {}
    for method in method_names:
        evals["auc_" + method] = roc_auc_score(y_true=scores_data_frame['label'], y_score=scores_data_frame[method])

    for model_type, graph_model in list(graph_models.items()):
        (loglik, aic, item_LLs) = graph_model.likelihoods(adj_mat, print_timing=print_timing)
        evals["loglikelihood_" + model_type] = loglik
        evals["akaike_" + model_type] = aic

    evals['constructAllPairsFromMDocs'] = adj_mat.shape[0]      # only correct when we're using all pairs
    evals['numPositives'] = scores_data_frame['label'].sum()
    evals['numAffils'] = adj_mat.shape[1]
    if want_exp_model:
        evals['expModelConverged'] = int(graph_models['exponential'].exp_model_converged)

    with open(evals_outfile, 'w') as fpout:
        print("Saving results to " + evals_outfile)
        for (measure, val) in sorted(evals.items()):
            fpout.write(measure + '\t' + str(val) + '\n')


def score_only(adj_mat_file, method_spec, pair_scores_outfile, flip_high_ps=False,
                 make_dense=True, row_labels=None, print_timing=False, learn_exp_model=False,
                 prefer_faiss=True, integer_ham_ssize=False):
    """

    :param adj_mat_file: function expects a file in matrix market format, optionally gzipped
    :param method_spec: list of method names (see scoring_methods.all_defined_methods for all choices)
    :param pair_scores_outfile: output path, should end in .csv.gz. Each line will contain one pair and all their scores.
    :param flip_high_ps:
    :param make_dense: If false, keep matrix in sparse format. Uses less RAM, but far slower.
    :param row_labels: Array of labels, in case 0:(num_rows(adj_mat)-1) isn't their usual naming/numbering
    :param print_timing:
    :param learn_exp_model: fit and compute likelihoods for exponential graph model even if not using it for scoring
    :param prefer_faiss: when the FAISS library is installed, use it (for the methods implemented in it)
    :param integer_ham_ssize: hamming (distance) and shared_size are returned as integers (saves space and easier to
                              interpret). The default changes them both to similarities between 0 and 1.
    :return: (no return value. instead, scores are saved to pair_scores_outfile.)
    """

    adj_mat = load_adj_mat(adj_mat_file)
    _, adj_mat, row_labels = remove_boundary_nodes(adj_mat, flip_high_ps=flip_high_ps, orig_row_labels=row_labels)
    if make_dense:
        adj_mat = adj_mat.toarray()

    want_exp_model = learn_exp_model or ('weighted_corr_exp' in method_spec) or \
                     ('weighted_corr_exp_faiss' in method_spec) or ('all' in method_spec)
    graph_models = learn_graph_models(adj_mat, bernoulli=True, exponential=want_exp_model)

    for model_type, graph_model in list(graph_models.items()):
        (loglik, aic, item_LLs) = graph_model.likelihoods(adj_mat)
        print("loglikelihood " + model_type + ": " + str(loglik))
        print("akaike " + model_type + ": " + str(aic))

    pairs_generator = partial(gen_all_pairs, row_labels=row_labels)    # this is a generator function. Call it with an arg to get generator object.
    indices_generator = partial(ij_gen, adj_mat.shape[0], row_labels)  # this too is a generator function. Call it w/o args to get generator object.

    scoring_methods.score_pairs(pairs_generator, adj_mat, method_spec, outfile_csv_gz=pair_scores_outfile,
                                pi_vector=graph_models['bernoulli'].affil_params,
                                indices_gen=indices_generator,
                                exp_model=graph_models.get('exponential', None),
                                num_docs=adj_mat.shape[0], mixed_pairs_sims='standard',
                                print_timing=print_timing, prefer_faiss=prefer_faiss, back_compat=integer_ham_ssize)
    print('scored pairs saved to ' + pair_scores_outfile)


def get_item_likelihoods(adj_mat_file, exponential_model=True, row_labels = None, adj_mat_ready = None):
    """
    Reads matrix, learns model of graph, returns vector of log likelihoods for items.
    :param adj_mat_file: in matrix market format, optionally compressed (*.mtx.gz)
    :param exponential_model: True for exponential model, False for Bernoulli model
    :param row_labels: optional vector of strings or numbers (item names)
    :param adj_mat_ready: Can pass in a sparse adjacency matrix instead of a file path -- see below.
    :return:
    """
    # allows input in the form of a file path OR an unweighted adj_mat
    if adj_mat_file == "" and adj_mat_ready is not None:
        adj_mat = adj_mat_ready
    else:
        adj_mat = load_adj_mat(adj_mat_file)

    pi_vector, adj_mat, row_labels = remove_boundary_nodes(adj_mat, orig_row_labels=row_labels)

    one_graph_model = learn_graph_models(adj_mat, bernoulli=(not exponential_model),
                                      pi_vector=pi_vector, exponential=exponential_model)
    (tot_loglik, aic, item_LLs) = list(one_graph_model.values())[0].likelihoods(adj_mat)
    print("learned " + list(one_graph_model.keys())[0] + " model. total loglikelihood " + str(tot_loglik) + ", aic " + str(aic))

    return item_LLs, row_labels


# Utility function: doesn't look at pairs, simply fits a model to the graph and prints the log likelihoods for each
# item. Runs both Bernoulli and exponential models.
def write_item_likelihoods(adj_mat_file, loglik_out_csv, flip_high_ps=False, row_labels = None):
    adj_mat = load_adj_mat(adj_mat_file)
    pi_vector, adj_mat, row_labels = remove_boundary_nodes(adj_mat, flip_high_ps=flip_high_ps,
                                                           orig_row_labels=row_labels)

    graph_models = learn_graph_models(adj_mat, bernoulli=True, pi_vector=pi_vector, exponential=True)

    (loglik_bern, aic_bern, item_LLs_bern) = graph_models['bernoulli'].likelihoods(adj_mat)
    (loglik_exp, aic_exp, item_LLs_exp) = graph_models['exponential'].likelihoods(adj_mat)
    print("bernoulli model: loglikelihood " + str(loglik_bern) + ", aic " + str(aic_bern))
    print("exponential model: loglikelihood " + str(loglik_exp) + ", aic " + str(aic_exp))

    with open(loglik_out_csv, 'w') as fout:
        fout.write("item,loglik_bernoulli,loglik_exponential\n")
        if row_labels is None:
            row_labels = range(adj_mat.shape[0])
        for i, score in enumerate(item_LLs_bern):
            fout.write(str(row_labels[i]) + ',' + str(item_LLs_bern[i]) + "," + str(item_LLs_exp[i]) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python score_data.py adj_matrix.mm.gz pair_scores_out.csv.gz method1 [method2 ...]")
        exit(0)

    datafile_mmgz = sys.argv[1]
    edge_scores_outfile = sys.argv[2]
    methods = [x for x in sys.argv[2:]]

    score_only(datafile_mmgz, methods, edge_scores_outfile)

