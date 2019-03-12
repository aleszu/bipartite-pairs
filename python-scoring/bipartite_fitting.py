
import bipartite_likelihood
import numpy as np
import time
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from collections import Counter


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
        indices_affils = range(adj_matrix.shape[0], adj_matrix.shape[0] + adj_matrix.shape[1]) * adj_matrix.shape[0]

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
    print 'Feature matrix constructed in {} seconds.'.format(round(t2 - t1), 2)
    t1 = time.time()

    # C is 1/regularization param. Set it high to effectively turn off regularization.
    # other param options: max_iter, solver, random_state (seed)
    log_reg_model = LogisticRegression(fit_intercept=use_intercept, C=1e15, solver='lbfgs').fit(X_feat, Y_lab)
    t2 = time.time()
    print 'Model learned in {} seconds.'.format(round(t2 - t1), 2) + " (" + str(log_reg_model.n_iter_[0]) + " iterations)"

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

    my_exp_model = bipartite_likelihood.exponentialModel(adj_matrix.shape[0], adj_matrix.shape[1])

    if use_intercept:
        intercept = log_reg_model.intercept_.squeeze()
        my_exp_model.set_density_param(intercept)

    if use_item_params:
        my_exp_model.set_item_params(model_coeffs[:adj_matrix.shape[0]])
    if use_affil_params:
        my_exp_model.set_affil_params(model_coeffs[adj_matrix.shape[0]:])

    return my_exp_model, model_ll


# straight from score_data.py
def learn_bernoulli(adj_matrix):
    pi_vector = adj_matrix.sum(axis=0) / float(adj_matrix.shape[0])
    return bipartite_likelihood.bernoulliModel(np.asarray(pi_vector).squeeze())


def learn_biment(adj_matrix, max_iter=5000):
    # from Albert
    item_degrees = np.asarray(adj_matrix.sum(axis=1)).squeeze()
    affil_degrees = np.asarray(adj_matrix.sum(axis=0)).squeeze()
    # temp change for testing!
    X, Y, X_bak, Y_bak = BiMent_solver(item_degrees, affil_degrees, tolerance=1e-5, max_iter=max_iter)
    #phi_ia = X[:, None] * Y / (1 + X[:, None] * Y)  # P(edge) matrix

    if X_bak is None and Y_bak is None:
        converge = False
    else:
        converge = True

    # the model is an exponential with item_param[i] = ln(X[i]) and affil_param[j] = ln(Y[j])
    expMod = bipartite_likelihood.exponentialModel(adj_matrix.shape[0], adj_matrix.shape[1])
    expMod.set_item_params(np.log(X))
    expMod.set_affil_params(np.log(Y))

    return expMod


# Single function taken from https://github.com/naviddianati/BiMent, as modified in
# https://github.com/agongt408/ASOUND. Then sped up by noting that every node with the same
# degree has the exact same param calc, so many of the calcs were redundant.
def BiMent_solver(fs, gs, tolerance=1e-10, max_iter=1000, first_order=False):
    '''
    Numerically solve the system of nonlinear equations
    we encounter when solving for the Lagrange multipliers
    @param fs: sequence of item degrees.
    @param gs: sequence of affiliation degrees.
    @param tolerance: solver continues iterating until the
    RMS of the difference between two consecutive solutions
    is less than tolerance.
    @param max_iter: maximum number of iterations.
    '''

    N = np.sum(fs)
    X_bak = fs / np.sqrt(N)     # store first approx in case that's what we return
    Y_bak = gs / np.sqrt(N)

    if first_order:
        return X_bak, Y_bak, None, None

    # set of unique degrees for each type of node
    # for either type of node, max number of unique degrees = min(len(fs), len(gs))
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
        pdeg_X[index] = deg / np.sqrt(N)     # initialize params to learn
        counts_deg_X[index] = cnt
        index += 1

    # ditto for Y and gs
    pdeg_Y = np.zeros(len(freqs_gs))
    counts_deg_Y = np.zeros(len(freqs_gs), dtype=np.int)
    index = 0
    for (deg, cnt) in sorted(freqs_gs.items()):
        pdeg_Y[index] = deg / np.sqrt(N)
        counts_deg_Y[index] = cnt
        index += 1

    change = 1
    t1 = time.time()
    for counter in np.arange(max_iter):
        pXYt = np.matmul(pdeg_X.reshape((-1, 1)), pdeg_Y.reshape((1, -1)))  # XYt[i,j] is x[i]*y[j]
        # matrix row of ( pdeg_y[j] / (1+pdeg_x[i]pdeg_y[j]) ) dotted with counts vector
        new_pdeg_x = sorted_degs_fs / (pdeg_Y / (1. + pXYt)).dot(counts_deg_Y)
        new_pdeg_y = sorted_degs_gs / (pdeg_X / (1. + pXYt.transpose())).dot(counts_deg_X)

        # for i in xrange(len(pdeg_X)):  # (untested, but something like this)
        #     pdeg_x[i] = sorted_degs_fs[i] / counts_deg_Y.dot(pdeg_Y / (1. + X[i] * pdeg_Y))
        # for j in xrange(len(pdeg_Y)):
        #     pdeg_y[j] = sorted_degs_gs[j] / counts_deg_X.dot(pdeg_X / (1. + Y[j] * pdeg_X))

        # L_oo
        change = max(np.max(np.abs(pdeg_X - new_pdeg_x)), np.max(np.abs(pdeg_Y - new_pdeg_y)))

        if counter % 500 == 0:
            print 'counter=%d, change=%f' % (counter, change)

        pdeg_X[:] = new_pdeg_x
        pdeg_Y[:] = new_pdeg_y
        if change < tolerance:
            break

    t2 = time.time()
    print 'Solver done in {} seconds.'.format(round(t2 - t1), 2)

    if change > tolerance:
        print "Warning: Solver did not converge. Returned first-order solution instead."
        return X_bak, Y_bak, None, None

    print "Solver converged in {} iterations.".format(counter)

    # convert back from deg_X (short) to X (longer)
    X = np.zeros(len(fs))
    Y = np.zeros(len(gs))
    for i in range(len(fs)):
        deg_node_i = fs[i]
        index_to_use_in_deg_X = sorted_degs_fs.index(deg_node_i)
        X[i] = pdeg_X[index_to_use_in_deg_X]

    for i in range(len(gs)):
        deg_node_i = gs[i]
        index_to_use_in_deg_Y = sorted_degs_gs.index(deg_node_i)
        Y[i] = pdeg_Y[index_to_use_in_deg_Y]

    return X, Y, X_bak, Y_bak
