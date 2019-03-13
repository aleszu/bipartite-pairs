from __future__ import print_function
from scipy.io import mmread
from scipy import sparse
import gzip
from sklearn.metrics import roc_auc_score
import sys
import numpy as np
import scoring_methods
import bipartite_fitting
import bipartite_likelihood


# If run from the command line, script simply reads data file, saves pairs with scores.
# Use run_and_eval() for expts: options for loading and manipulating phi, know true pairs and compute AUCs.

# Iterates through all pairs of matrix rows, in the form (i, j) where i < j
# Rows output are always of type numpy.ndarray()
# Returns (row1_index, row2_index, row1_label (differs from row1_index if row_labels param sent), row2_label, row1, row2)
# row "labels" are actually integers
def gen_all_pairs(my_adj_mat, row_labels=None):
    num_rows = my_adj_mat.shape[0]
    is_sparse = sparse.isspmatrix(my_adj_mat)
    is_numpy_matrix = (type(my_adj_mat) == np.matrix)
    for i in range(num_rows):
        if is_sparse:
            rowi = my_adj_mat.getrow(i).toarray()[0]  # toarray() gives 2-d matrix, [0] to flatten
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


# Infile must be in "matrix market" format and gzipped
# Returns a "compressed sparse column" matrix (for efficient column slicing)
def load_adj_mat(datafile_mmgz, make_binary = True):
    with gzip.open(datafile_mmgz, 'r') as fp_mm:
        adj_mat = mmread(fp_mm).astype(int, copy=False)  # creates a scipy.sparse.coo_matrix, if matrix was sparse
    if make_binary:     # change edge weights to 1
        adj_mat = adj_mat.astype(bool, copy=False).astype(int, copy=False)

    print("Read data: " + str(adj_mat.shape[0]) + " items, " + str(adj_mat.shape[1]) + " affiliations")
    return adj_mat.tocsc()


def load_pi_from_file(pi_vector_infile_gz):
    pi_saved = []
    with gzip.open(pi_vector_infile_gz, 'r') as fpin:
        for line in fpin:
            pi_saved.append(float(line.strip()))
    return np.array(pi_saved)  # making type consistent with learn_pi_vector


# Returns a numpy.ndarray with 1 axis (.ndim = 1)
def learn_pi_vector(adj_mat):
    pi_orig = adj_mat.sum(axis=0) / float(adj_mat.shape[0])  # a numpy matrix, a type that always stays 2D unless we call .tolist()[0] or np.asarray()
    return np.asarray(pi_orig).squeeze()


# Always: remove exact 0s and 1s from columns of data + pi_vector.
# Optionally: "flip" high p's -- i.e., swap 1's and 0's in the data so that resulting p's are <= .5.
# expt1: remove affils with 0 or even 1 person attached
def adjust_pi_vector(pi_vector, adj_mat, flip_high_ps=False, expt1 = False):
    epsilon = .25 / adj_mat.shape[0]  # If learned from the data, p_i would be in increments of 1/nrows
    if expt1:
        print("expt1: removing affils with degree 0 *or 1*")
        affils_to_keep = np.logical_and(pi_vector >= epsilon + float(1)/adj_mat.shape[0],
                                        pi_vector <= 1 - epsilon - float(1)/adj_mat.shape[0])
    else:
        affils_to_keep = np.logical_and(pi_vector >= epsilon, pi_vector <= 1 - epsilon)
    print("Keeping " + ("all " if (affils_to_keep.sum() == adj_mat.shape[0]) else "") \
          + str(affils_to_keep.sum()) + " affils")
    which_nonzero = np.nonzero(affils_to_keep)      # returns a tuple (immutable list) holding 1 element: an ndarray of indices
    pi_vector_mod = pi_vector[which_nonzero]        # since pi_vector is also an ndarray, the slicing is happy to use a tuple
    adj_mat_mod = adj_mat[:, which_nonzero[0]]      # since adj_mat is a matrix, slicing needs just the ndarray

    cmpts_to_flip = pi_vector_mod > .5
    if flip_high_ps:
        print("Flipping " + str(cmpts_to_flip.sum()) + " components that had p_i > .5")
        print("(ok to ignore warning message produced)")
        which_nonzero = np.nonzero(cmpts_to_flip)
        pi_vector_mod[which_nonzero] = 1 - pi_vector_mod[which_nonzero]
        adj_mat_mod[:, which_nonzero[0]] = np.ones(adj_mat_mod[:, which_nonzero[0]].shape, dtype=adj_mat_mod.dtype) \
                                           - adj_mat_mod[:, which_nonzero[0]]

    else:
        print("fyi: leaving in the " + str(cmpts_to_flip.sum()) + " components with p_i > .5")

    return pi_vector_mod, adj_mat_mod.tocsr()


# Convention for all my expt data: the true pairs are items (1,2), (3,4), etc.
def get_true_labels_expt_data(pairs_generator, num_true_pairs):
    labels = []
    for (row_idx1, row_idx2, _, _, pair_x, pair_y) in pairs_generator:
        label = True if (row_idx2 < 2 * num_true_pairs and row_idx1 == row_idx2 - 1 and row_idx2 % 2) else False
        labels.append(label)
    return labels


def true_labels_for_expts_with_5pairs(pairs_generator):
    return get_true_labels_expt_data(pairs_generator, 5)

# (max_iter_biment moved here to be easier to change. we did hit ~51k iterations for one matrix, dims 969 x 42k)
def learn_graph_models(adj_mat, bernoulli=True, pi_vector=None, exponential=False, max_iter_biment=5000):
    graph_models = dict()
    if bernoulli:
        if pi_vector is not None:
            bernoulli = bipartite_likelihood.bernoulliModel(pi_vector)
        else:
            bernoulli = bipartite_fitting.learn_bernoulli(adj_mat)
        graph_models['bernoulli'] = bernoulli
    if exponential:
        graph_models['exponential'] = bipartite_fitting.learn_biment(adj_mat, max_iter=max_iter_biment)
    return graph_models


def run_and_eval(adj_mat, true_labels_func, method_spec, evals_outfile,
                 pair_scores_outfile=None, pi_vector_infile=None, flip_high_ps=False,
                 make_dense=True, row_labels=None, print_timing=False, expt1=False, learn_exp_model=False,
                 prefer_faiss=False):
    """

    :param adj_mat:
    :param true_labels_func: identifies the true pairs, given a pairs_generator
    :param method_spec: list of method names OR the string 'all'
    :param evals_outfile:
    :param pair_scores_outfile:
    :param pi_vector_infile:
    :param flip_high_ps:
    :param make_dense:
    :param row_labels: used when adj_mat's indices differ from original row numbers
    :param print_timing:
    :param expt1:
    :param learn_exp_model:
    :return:
    """

    # note on sparse matrices: adj_mat is initially read in as "coo" format (coordinates of entries). Next few operations
    # will be by column, so it's returned from load_adj_mat as "csc" (compressed sparse column). Then, converted to
    # "csr" in adjust_pi_vector to make pair generation (row slicing) fast.

    # learn phi (/pi_vector)
    if pi_vector_infile is not None:
        pi_vector = load_pi_from_file(pi_vector_infile)
    else:
        pi_vector = learn_pi_vector(adj_mat)

    pi_vector, adj_mat = adjust_pi_vector(pi_vector, adj_mat, flip_high_ps, expt1=expt1)
    if make_dense:
        adj_mat = adj_mat.toarray()

    # score pairs
    # (sending in all special args any methods might need)

    want_exp_model = learn_exp_model or ('weighted_corr_exp' in method_spec) or\
                     ('weighted_corr_exp_faiss' in method_spec) or ('all' in method_spec)
    graph_models = learn_graph_models(adj_mat, bernoulli=True, pi_vector=pi_vector, exponential=want_exp_model)

    # First, run any methods that return a subset of pairs (right now, none -- expect to need this when scaling up).
    # scores_subset =
    # Once implemented, make pairs_generator use the pairs in scores_subset.

    if row_labels is None:
        pairs_generator = gen_all_pairs
    else:
        def my_pairs_gen(adj_mat):
            return gen_all_pairs(adj_mat, row_labels)
        pairs_generator = my_pairs_gen

    scores_data_frame = scoring_methods.score_pairs(pairs_generator, adj_mat, method_spec,
                                                    pi_vector=pi_vector, num_docs=adj_mat.shape[0],
                                                    mixed_pairs_sims = 'standard',
                                                    exp_model=graph_models.get('exponential', None),
                                                    print_timing=print_timing,
                                                    prefer_faiss=prefer_faiss)
    # if scores_subset is not None:
    #     scores_data_frame = pd.merge(scores_subset, scores_data_frame, on=['item1', 'item2'])

    method_names = set(scores_data_frame.columns.tolist()) - {'item1', 'item2'}
    scores_data_frame['label'] = map(int, true_labels_func(pairs_generator(adj_mat)))

    # round pair scores at 15th decimal place so we don't get spurious diffs in AUCs when replicating
    scores_data_frame = scores_data_frame.round(decimals={method:15 for method in method_names})

    # save pair scores if desired
    if pair_scores_outfile is not None:
        scores_data_frame = scores_data_frame.reindex(columns=['item1', 'item2', 'label'] +
                                                              sorted(list(method_names - {'label'})), copy=False)
        scores_data_frame.to_csv(pair_scores_outfile, index=False, compression="gzip")

    # compute evals and save
    evals = {}
    for method in method_names:
        evals["auc_" + method] = roc_auc_score(y_true=scores_data_frame['label'], y_score=scores_data_frame[method])

    for model_type, graph_model in graph_models.items():
        evals["loglikelihood_" + model_type] = graph_model.loglikelihood(adj_mat)
        evals["akaike_" + model_type] = graph_model.akaike(adj_mat)

    evals['constructAllPairsFromMDocs'] = adj_mat.shape[0]      # only correct when we're using all pairs
    evals['numPositives'] = scores_data_frame['label'].sum()
    evals['numAffils'] = adj_mat.shape[1]

    with open(evals_outfile, 'w') as fpout:
        print("Saving results to " + evals_outfile)
        for (measure, val) in sorted(evals.iteritems()):
            fpout.write(measure + '\t' + str(val) + '\n')


def score_only(adj_mat_file, method_spec, pair_scores_outfile, flip_high_ps=False,
                 make_dense=True, row_labels=None, print_timing=False, learn_exp_model=False,
                 prefer_faiss=True):

    adj_mat = load_adj_mat(adj_mat_file)
    pi_vector = learn_pi_vector(adj_mat)
    pi_vector, adj_mat = adjust_pi_vector(pi_vector, adj_mat, flip_high_ps)
    if make_dense:
        adj_mat = adj_mat.toarray()

    want_exp_model = learn_exp_model or ('weighted_corr_exp' in method_spec) or \
                     ('weighted_corr_exp_faiss' in method_spec) or ('all' in method_spec)
    graph_models = learn_graph_models(adj_mat, bernoulli=True, pi_vector=pi_vector, exponential=want_exp_model)

    for model_type, graph_model in graph_models.items():
        print("loglikelihood " + model_type + ": " + str(graph_model.loglikelihood(adj_mat)))
        print("akaike " + model_type + ": " + str(graph_model.akaike(adj_mat)))

    # First, run any methods that return a subset of pairs (right now, none -- expect to need this when scaling up).
    # scores_subset =
    # Once implemented, make pairs_generator use the pairs in scores_subset.

    if row_labels is None:
        pairs_generator = gen_all_pairs
    else:
        def my_pairs_gen(adj_mat):
            return gen_all_pairs(adj_mat, row_labels)

        pairs_generator = my_pairs_gen

    scores_data_frame = scoring_methods.score_pairs(pairs_generator, adj_mat, method_spec,
                                                    pi_vector=pi_vector, num_docs=adj_mat.shape[0],
                                                    mixed_pairs_sims='standard',
                                                    exp_model=graph_models.get('exponential', None),
                                                    print_timing=print_timing,
                                                    prefer_faiss=prefer_faiss)

    # save results
    method_names = set(scores_data_frame.columns.tolist()) - {'item1', 'item2'}
    scores_data_frame = scores_data_frame.reindex(columns=['item1', 'item2'] + sorted(method_names), copy=False)
    scores_data_frame.to_csv(pair_scores_outfile, index=False, compression="gzip")
    print('scored pairs saved to ' + pair_scores_outfile)


# __main__ function should take: infile (adj_matrix.mm.gz), outfile (edge scores), methods (space-separated list of
# scoring methods to run)
# (todo? implement a way to specify param values for MixedPairs at command line)
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python score_data.py adj_matrix.mm.gz pair_scores_out.csv.gz method1 [method2 ...]")
        exit(0)

    datafile_mmgz = sys.argv[1]
    edge_scores_outfile = sys.argv[2]
    methods = [x for x in sys.argv[2:]]
    # if 'all' in methods:   # moved into score_pairs
    #     methods = scoring_methods.all_defined_methods

    # infile --> adj matrix
    adj_mat = load_adj_mat(datafile_mmgz)

    # learn phi
    pi_vector = learn_pi_vector(adj_mat)
    pi_vector, adj_mat = adjust_pi_vector(pi_vector, adj_mat)

    # score pairs
    scores_data_frame = scoring_methods.score_pairs(gen_all_pairs, adj_mat, methods, pi_vector=pi_vector)

    # save results
    method_names = set(scores_data_frame.columns.tolist()) - {'item1', 'item2'}
    scores_data_frame = scores_data_frame.reindex(columns=['item1', 'item2'] + sorted(method_names), copy=False)
    scores_data_frame.to_csv(edge_scores_outfile, index=False, compression="gzip")
