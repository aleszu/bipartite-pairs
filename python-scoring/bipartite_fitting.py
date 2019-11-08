from __future__ import print_function
from builtins import str, range
import bipartite_likelihood
import numpy as np
import time
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from collections import Counter


# (max_iter_biment moved here to be easier to change. we did hit ~51k iterations for one matrix, dims 969 x 42k)
def learn_graph_models(adj_mat, bernoulli=True, pi_vector=None, exponential=False, max_iter_biment=5000, verbose=False):
    graph_models = dict()
    if bernoulli:
        if pi_vector is not None:
            bernoulli = bipartite_likelihood.bernoulliModel(pi_vector)
        else:
            bernoulli = learn_bernoulli(adj_mat)
        graph_models['bernoulli'] = bernoulli
    if exponential:
        graph_models['exponential'] = learn_biment(adj_mat, max_iter=max_iter_biment, verbose=verbose)
    return graph_models


# straight from score_data.py
def learn_bernoulli(adj_matrix):
    pi_vector = adj_matrix.sum(axis=0) / float(adj_matrix.shape[0])
    return bipartite_likelihood.bernoulliModel(np.asarray(pi_vector).squeeze())


def learn_biment(adj_matrix, max_iter=5000, verbose=False):

    item_degrees = np.asarray(adj_matrix.sum(axis=1)).squeeze()
    affil_degrees = np.asarray(adj_matrix.sum(axis=0)).squeeze()
    X, Y, X_bak, Y_bak = BiMent_solver(item_degrees, affil_degrees, tolerance=1e-5, max_iter=max_iter, verbose=verbose)
    #phi_ia = X[:, None] * Y / (1 + X[:, None] * Y)  # P(edge) matrix

    # the model is an exponential with item_param[i] = ln(X[i]) and affil_param[j] = ln(Y[j])
    expMod = bipartite_likelihood.ExponentialModel(adj_matrix.shape[0], adj_matrix.shape[1])
    with np.errstate(divide='ignore'):  # don't warn for log(0) (= -inf)
        expMod.set_item_params(np.log(X))
        expMod.set_affil_params(np.log(Y))
    if X_bak is not None:
        expMod.exp_model_converged = True

    return expMod


# Single function taken from https://github.com/naviddianati/BiMent, as modified in
# https://github.com/agongt408/ASOUND. Then sped up by noting that every node with the same
# degree has the exact same param calc, so many of the calcs were redundant.
def BiMent_solver(fs, gs, tolerance=1e-10, max_iter=1000, first_order=False, verbose=True,
                           simple_bdry_handling=False, anchor_one_param=True):
    '''
    Numerically solve the system of nonlinear equations we encounter when solving for the Lagrange multipliers
    @param fs: sequence of item degrees.
    @param gs: sequence of affiliation degrees.
    @param tolerance: solver continues iterating until the RMS of the difference between two consecutive solutions
        is less than tolerance.
    @param max_iter: maximum number of iterations.
    @param first_order: don't solve, just return initializations
    @param verbose:
    @param simple_bdry_handling: Note that the model's MLE is not defined when any node has degree = 0 or max! However,
        the math works out(*) if we simply give it param 0 or infinity, respectively (and adjust the code to carry out
        these computations without warnings or errors).
        Even without special handling, nodes with degree 0 converge to param = 0.
        But nodes that have every possible connection cause problems for fitting. We want to give them parameter = infinity.
        If True (default), do this the simplest way, manually setting it (and excluding them from the solver).
        If False, we initialize the param to a high number, and wait for the alg to converge.

        (*) However, the workaround fails for graphs in which deleting a degree-0 node creates a (induced) max-degree
        node. In that case, there's no MLE even if you allow infinities (because we don't have a way to assign precedence
        between the 0 param and the infinity). BiMent_solver() will not converge for such graphs.
        (**) For graphs where deleting a max-degree node creates a (induced) 0-degree node, the same conflict exists, in
        theory, but in practice, the algorithm will converge, giving a near-0 parameter to the 0-degree node. (Close enough!)

    @param anchor_one_param: The model's params aren't fully identifiable: you can increase all item params by
        delta and decrease all affil params by the same delta, without affecting the likelihoods. Here, ensure the params
        themselves come out identical, when they're equivalent, by setting the lowest-degree
        item to have param X = 1 (item_param = log(X) = 0) and adjusting the others accordingly.
    '''

    N = np.sum(fs)
    X_bak = fs / np.sqrt(N)     # store first approx in case that's what we return
    Y_bak = gs / np.sqrt(N)

    if first_order:
        return X_bak, Y_bak, None, None

    # set of unique degrees for each type of node
    if simple_bdry_handling:
        # special handling for maximum-degree boundary nodes: pretend we don't see them
        # (note: this approach isn't a general solution, b/c behaves incorrectly when deleting a boundary node
        # leaves another with degree 0. this code then gives that node param 0, which is wrong. test case 6 fails.)
        num_max_fs = Counter(fs)[len(gs)]
        num_max_gs = Counter(gs)[len(fs)]
        fs_masked = fs[fs < len(gs)]
        gs_masked = gs[gs < len(fs)]
        # make sure to remove all traces
        N = np.sum(fs[fs < len(gs)])  # edges in graph
        if num_max_fs > 0:  # when deleting a node in g, decrement degrees of all nodes in f
            gs_masked = gs_masked - num_max_fs
        if num_max_gs > 0:
            fs_masked = fs_masked - num_max_gs

        freqs_fs = Counter(fs_masked)
        freqs_gs = Counter(gs_masked)

    else:
        freqs_fs = Counter(fs)
        freqs_gs = Counter(gs)

    sorted_degs_fs = sorted(freqs_fs)
    sorted_degs_gs = sorted(freqs_gs)

    # Set up data structures:
    # -pdeg_X: params as a function of node degree. one param per unique node degree. Replaces X.
    # -counts_deg_X (same length as deg_X): for each degree, how many nodes have it?
    pdeg_X = np.zeros(len(freqs_fs))
    counts_deg_X = np.zeros(len(freqs_fs), dtype=np.int)
    index = 0
    for (deg, cnt) in sorted(freqs_fs.items()):
        pdeg_X[index] = deg / np.sqrt(N)    # initialize params to learn
        if deg == len(gs):                  # special case for "boundary" nodes connected to everyone
            pdeg_X[index] = np.inf          # or np.finfo(type(pdeg_X[index])).max / 2 or similarly big
        counts_deg_X[index] = cnt
        index += 1

    # ditto for Y and gs
    pdeg_Y = np.zeros(len(freqs_gs))
    counts_deg_Y = np.zeros(len(freqs_gs), dtype=np.int)
    index = 0
    for (deg, cnt) in sorted(freqs_gs.items()):
        pdeg_Y[index] = deg / np.sqrt(N)
        if deg == len(fs):
            pdeg_Y[index] = np.inf

        counts_deg_Y[index] = cnt
        index += 1

    change = 1
    t1 = time.time()
    for counter in np.arange(max_iter):
        pXYt = np.matmul(pdeg_X.reshape((-1, 1)), pdeg_Y.reshape((1, -1)))  # XYt[i,j] is x[i]*y[j]

        # Edge probs are XY / (1 + XY)
        # Expected degrees for nodes in f: for each row, its exp_deg = row(edge probs).dot(counts_deg_Y)
        # Expected degrees for nodes in g: for each col, its exp_deg = col(edge probs).dot(counts_deg_X)
        # Compute: new_param_for_degree = empirical degree / (expected degree / old_param_for_degree)

        # (non-vector version would be something like this)
        # for i in xrange(len(pdeg_X)):
        #     pdeg_x[i] = sorted_degs_fs[i] / counts_deg_Y.dot(pdeg_Y / (1. + X[i] * pdeg_Y))
        # for j in xrange(len(pdeg_Y)):
        #     pdeg_y[j] = sorted_degs_gs[j] / counts_deg_X.dot(pdeg_X / (1. + Y[j] * pdeg_X))

        if False and simple_bdry_handling:
            # last term is matrix row of ( pdeg_y[j] / (1+pdeg_x[i]pdeg_y[j]) ) dotted with counts vector
            new_pdeg_x = sorted_degs_fs / (pdeg_Y / (1. + pXYt)).dot(counts_deg_Y)
            new_pdeg_y = sorted_degs_gs / (pdeg_X / (1. + pXYt.transpose())).dot(counts_deg_X)

        else:        # modified to avoid nan, which occurs whenever XY = inf
            # for new_pdeg_x:
            #   When Y == inf, we want XY / (1+XY) = P(edge) = 1 (instead of nan), so use Y/(1+XY) = 1/X
            #   (But when X == inf, need the usual new_pdeg_x calc, provided we allow calculating x / 0 --> inf.)
            # (np.where() is looking at a matrix)
            with np.errstate(divide='ignore', invalid='ignore'):    # don't warn about 1/0 or inf/inf
                new_pdeg_x = sorted_degs_fs / np.where(np.isinf(pdeg_Y.reshape((1,-1))), (1 / pdeg_X).reshape((-1,1)),
                                                       (pdeg_Y / (1. + pXYt)) ).dot(counts_deg_Y)
                new_pdeg_y = sorted_degs_gs / np.where(np.isinf(pdeg_X.reshape((1,-1))), (1 / pdeg_Y).reshape((-1,1)),
                                                       (pdeg_X / (1. + pXYt.transpose())) ).dot(counts_deg_X)

        # L_oo
        change = max(np.max(np.abs(np.nan_to_num(pdeg_X) - np.nan_to_num(new_pdeg_x))),
                     np.max(np.abs(np.nan_to_num(pdeg_Y) - np.nan_to_num(new_pdeg_y))))

        if verbose and counter % 500 == 0:
            print('counter=%d, change=%f' % (counter, change))

        pdeg_X[:] = new_pdeg_x
        pdeg_Y[:] = new_pdeg_y
        if change < tolerance:
            break

    t2 = time.time()
    if verbose:
        print('Solver done in {} seconds.'.format(round(t2 - t1), 2))

    if change > tolerance:
        print("Warning: Solver did not converge after {} iterations. Returned first-order solution instead.".format(counter))
        return X_bak, Y_bak, None, None

    if verbose:
        print("Solver converged in {} iterations.".format(counter))

    if anchor_one_param:
        # set pdeg_X[0] = 1, and everyone else adjusts accordingly.
        # But if there's a node with deg 0, leave it alone and anchor pdeg_X[1] instead.
        min_idx_pos_deg = np.min(np.nonzero(sorted_degs_fs))
        # note that the delta is in the exponent (exp^(alpha + beta + delta)); since X = e^alpha, that translates to
        # multiplication or division here.
        delta = pdeg_X[min_idx_pos_deg]
        pdeg_X[min_idx_pos_deg:] = pdeg_X[min_idx_pos_deg:] / delta
        pdeg_Y = pdeg_Y * delta

    # convert back from deg_X (short) to X (longer)
    X = np.zeros(len(fs))
    Y = np.zeros(len(gs))
    for i in range(len(fs)):
        true_deg_node_i = fs[i]
        if simple_bdry_handling:
            if true_deg_node_i == 0:
                X[i] = 0
            elif true_deg_node_i == len(gs):
                X[i] = np.inf
            else:
                index_to_use_in_deg_X = sorted_degs_fs.index(true_deg_node_i - num_max_gs)
                X[i] = pdeg_X[index_to_use_in_deg_X]
        else:
            index_to_use_in_deg_X = sorted_degs_fs.index(true_deg_node_i)
            X[i] = pdeg_X[index_to_use_in_deg_X]

    for i in range(len(gs)):
        true_deg_node_i = gs[i]
        if simple_bdry_handling:
            if true_deg_node_i == 0:
                Y[i] = 0
            elif true_deg_node_i == len(fs):
                Y[i] = np.inf
            else:
                index_to_use_in_deg_Y = sorted_degs_gs.index(true_deg_node_i - num_max_fs)
                Y[i] = pdeg_Y[index_to_use_in_deg_Y]
        else:
            index_to_use_in_deg_Y = sorted_degs_gs.index(true_deg_node_i)
            Y[i] = pdeg_Y[index_to_use_in_deg_Y]

    return X, Y, X_bak, Y_bak


# Defaults to the full model; have to turn off unwanted params explicitly.
def learn_exponential_model(adj_matrix, use_intercept=True, use_item_params=True, use_affil_params=True,
                            withLL = False):

    # Create feature matrix and labels for the logistic regression
    # Iterate through entries of the adj_matrix, which become our instances. (Neither matrix is symmetric.)

    # for sklearn.linear_model.LogisticRegression, input should be sparse CSR

    t1 = time.time()
    # naive version, very slow
    # X_feat = sparse.csr_matrix((np.product(adj_matrix.shape), np.sum(adj_matrix.shape)), dtype=bool)
    # Y_lab = np.zeros(np.product(adj_matrix.shape), dtype=bool)
    # inst_num = 0
    # for i in range(adj_matrix.shape[0]):
    #     for j in range(adj_matrix.shape[1]):
    #         X_feat[inst_num, i] = use_item_params  # instance inst_num refers to item i
    #         X_feat[inst_num, adj_matrix.shape[0] + j] = use_affil_params   # instance inst_num refers to affil j
    #         Y_lab[inst_num] = adj_matrix[i,j]   # edge presence/absence
    #         inst_num += 1


    # Version 2 of matrix construction. indptr tells where the new rows start in the vectors "indices" and "data".
    # indices stores column indices.
    Y_lab = adj_matrix.reshape((-1,1)).toarray().squeeze().astype(bool)
    if use_item_params:
        indices_items = np.concatenate([np.full(shape=adj_matrix.shape[1], fill_value=x) for x in range(adj_matrix.shape[0])])
    if use_affil_params:
        indices_affils = list(range(adj_matrix.shape[0], adj_matrix.shape[0] + adj_matrix.shape[1])) * adj_matrix.shape[0]

    if use_item_params and use_affil_params:
        # X_feat will have 2 entries per row. Num rows = np.product(adj_matrix.shape).
        # data and indices will be length 2 * np.product(adj_matrix.shape).
        # indptr's last entry is their length, but np.arange() needs to be told 1 more than that.
        indptr = np.arange(start=0, stop=2 * np.product(adj_matrix.shape) + 1, step=2)
        data = np.full(shape=2 * np.product(adj_matrix.shape), fill_value=True)
        indices = np.column_stack([indices_items, indices_affils]).reshape(-1)
        X_feat = sparse.csr_matrix((data, indices, indptr), dtype=bool)
    elif use_item_params or use_affil_params:
        # X_feat will have one entry per row
        indptr = np.arange(start=0, stop=np.product(adj_matrix.shape) + 1, step=1)
        data = np.full(shape=np.product(adj_matrix.shape), fill_value=True)
        # which entry?
        if use_item_params:
            indices = indices_items
        elif use_affil_params:
            indices = indices_affils
        X_feat = sparse.csr_matrix((data, indices, indptr), dtype=bool)
    else:
        X_feat = sparse.csr_matrix((np.product(adj_matrix.shape), np.sum(adj_matrix.shape)), dtype=bool)

    t2 = time.time()
    print('Feature matrix constructed in {} seconds.'.format(round(t2 - t1), 2))
    t1 = time.time()

    # C is 1/regularization param. Set it high to effectively turn off regularization.
    # other param options: max_iter, solver, random_state (seed)
    log_reg_model = LogisticRegression(fit_intercept=use_intercept, C=1e15, solver='lbfgs').fit(X_feat, Y_lab)
    t2 = time.time()
    print('Model learned in {} seconds.'.format(round(t2 - t1), 2) + " (" + str(log_reg_model.n_iter_[0]) + " iterations)")

    # for fun & sanity checking, get its predictions back out
    if withLL:
        preds = log_reg_model.predict_log_proba(X_feat)  # P(True) will always be in 2nd col
        which_nonzero = np.nonzero(Y_lab)
        which_zero = np.nonzero(np.logical_not(Y_lab))
        model_ll = np.sum(preds[which_nonzero,1]) + np.sum(preds[which_zero, 0])
        # print "log likelihood of input data according to sklean model: " + str(model_ll)
    else:
        model_ll = None

    model_coeffs = log_reg_model.coef_.squeeze()  # 1 per column of X_feat

    my_exp_model = bipartite_likelihood.ExponentialModel(adj_matrix.shape[0], adj_matrix.shape[1])

    if use_intercept:
        intercept = log_reg_model.intercept_.squeeze()
        my_exp_model.set_density_param(intercept)

    if use_item_params:
        my_exp_model.set_item_params(model_coeffs[:adj_matrix.shape[0]])
    if use_affil_params:
        my_exp_model.set_affil_params(model_coeffs[adj_matrix.shape[0]:])

    return my_exp_model, model_ll
