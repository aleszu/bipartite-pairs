
import bipartite_likelihood
import numpy as np
import time
from scipy import sparse
from sklearn.linear_model import LogisticRegression



# Defaults to the full model; have to turn off unwanted params explicitly.
def learn_exponential_model(adj_matrix, use_intercept=True, use_item_params=True, use_affil_params=True):

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
    preds = log_reg_model.predict_log_proba(X_feat)  # P(True) will always be in 2nd col
    which_nonzero = np.nonzero(Y_lab)
    which_zero = np.nonzero(np.logical_not(Y_lab))
    model_ll = np.sum(preds[which_nonzero,1]) + np.sum(preds[which_zero, 0])
    # print "log likelihood of input data according to sklean model: " + str(model_ll)

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


def learn_biment(adj_matrix):
    # from Albert
    item_degrees = np.asarray(adj_matrix.sum(axis=1)).squeeze()
    affil_degrees = np.asarray(adj_matrix.sum(axis=0)).squeeze()
    X, Y, X_bak, Y_bak = BiMent_solver(item_degrees, affil_degrees, tolerance=1e-5, max_iter=5000)
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
# https://github.com/agongt408/ASOUND
def BiMent_solver(fs, gs, tolerance=1e-10, max_iter=1000, first_order=False):
    '''
    Numerically solve the system of nonlinear equations
    we encounter when solving for the Lagrange multipliers
    @param fs: sequence of symbol frequencies.
    @param gs: sequence of set sizes.
    @param tolerance: solver continues iterating until the
    RMS of the difference between two consecutive solutions
    is less than tolerance.
    @param max_iter: maximum number of iterations.
    '''
    n, m = len(fs), len(gs)

    N = np.sum(fs)
    X = fs / np.sqrt(N)     # initialize params to learn
    Y = gs / np.sqrt(N)

    if first_order:
        return X, Y, None, None

    X_bak = np.copy(X)      # store initial values to return
    Y_bak = np.copy(Y)

    x = fs * 0             # new values set in loop
    y = gs * 0

    change = 1
    t1 = time.time()
    for counter in np.arange(max_iter):
        XYt = np.matmul(X.reshape((-1,1)), Y.reshape((1,-1)))  # XYt[i,j] is x[i]*y[j]
        x = fs / np.sum(Y / (1. + XYt), axis=1)                # np.sum( y[j] / (1+x[i]y[j]) )
        y = gs / np.sum(X / (1. + XYt.transpose()), axis=1)    # np.sum( x[j] / (1+x[j]y[i]) )

        # for i in xrange(n):
        #     x[i] = fs[i] / np.sum(Y / (1. + X[i] * Y))
        # for i in xrange(m):
        #     y[i] = gs[i] / np.sum(X / (1. + X * Y[i]))

        # L_oo
        change = max(np.max(np.abs(X - x))  , np.max(np.abs(Y - y)))

        if counter % 500 == 0:
            print 'counter=%d, change=%f' % (counter, change)

        X[:] = x
        Y[:] = y
        if change < tolerance:
            break

    t2 = time.time()
    print 'Solver done in {} seconds.'.format(round(t2 - t1), 2)

    if change > tolerance:
        print "Warning: Solver did not converge. Returned first-order solution instead."
        return X_bak, Y_bak, None, None

    print "Solver converged in {} iterations.".format(counter)
    return X, Y, X_bak, Y_bak