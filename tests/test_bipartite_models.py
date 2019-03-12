import score_data
import bipartite_likelihood
import bipartite_fitting
import numpy as np
import loc_data
from timeit import default_timer as timer
import sys
sys.path.append("../python-scoring")  # add other dirs to path (for non-PyCharm use)
sys.path.append("../expt-code")



def test_create_models():
    """
    Tries out constructing bipartiteGraphModel objects (with pre-specified params), checks that loglikelihood & akaike
     methods match my calculations (here).
    """
    print "\n*** Testing the bipartiteGraphModel class ***\n"
    # Using data we've already played with: take its pi_vector, put it into a bernoulliModel,
    # and use model to score the adj_matrix

    adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz"
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    pi_vector_learned = score_data.learn_pi_vector(adj_mat)
    pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

    bernoulli_class = bipartite_likelihood.bernoulliModel(pi_vector_preproc)
    print "bernoulli model: num_params is " + str(bernoulli_class.get_num_params())
    assert (bernoulli_class.get_num_params() == len(pi_vector_preproc))

    # look carefully at row 0 of matrix
    row0 = adj_mat_preproc.getrow(0).toarray()[0]
    print "row 0 has " + str(row0.sum()) + " non-zero entries, with the following p_i values:"
    which_nonzero = np.nonzero(row0)
    print str(pi_vector_preproc[which_nonzero])
    print "in log form: " + str(np.log(pi_vector_preproc[which_nonzero]))

    score_pos_edges0 = np.sum(np.log(pi_vector_preproc[which_nonzero]))
    score_pos_edges1 = row0.dot(np.log(pi_vector_preproc))
    print "that sum, two different ways: " + str(score_pos_edges0) + " or " + str(score_pos_edges1)

    which_zero = np.nonzero(1 - row0)
    score_neg_edges = (1 - row0).dot(np.log(1 - pi_vector_preproc))
    print "row 0 has " + str((1 - row0).sum()) + " zero entries, with p_i values that start out as:"
    print str(pi_vector_preproc[which_zero][:9])
    print "sum of all the log(1-p_i) entries:" + str(score_neg_edges)
    row0_likelihood = score_pos_edges1 + score_neg_edges

    # compare to what the model gives for first row of the adj_matrix
    model_ll = bernoulli_class.loglikelihood(adj_mat_preproc[0,])
    print "checking our log likelihood for row0 against the Bernoulli model's, which is " + str(model_ll)
    assert (abs(model_ll - row0_likelihood) < 1e-07)
    # check if aikake looks right, next to it. (for row0.)
    assert (abs(bernoulli_class.akaike(adj_mat_preproc[0,]) - 2 * (len(pi_vector_preproc) - row0_likelihood)) < 1e-07)
    print "akaike for row0: " + str(bernoulli_class.akaike(adj_mat_preproc[0,]))


    # can also test the exponential one in a trivial way by giving each edge equal density -- i.e., have it be Erdos-Renyi
    er_model = bipartite_likelihood.exponentialModel(adj_mat_preproc.shape[0], adj_mat_preproc.shape[1])
    er_model.set_density_param(-2)  # for all edges, p(edge) = e^-2 / 1 + e^-2, or approx 0.1192
    assert(er_model.get_num_params() == 1)
    print "ER model: num_params is 1"
    edge_prob = np.exp(-2) / (1 + np.exp(-2))
    our_er_loglik = np.sum(adj_mat_preproc) * np.log(edge_prob) + \
                    (np.product(adj_mat_preproc.shape) - np.sum(adj_mat_preproc)) * np.log(1 - edge_prob)
    print "checking our log likelihood for the ER graph against the model's, which is " + str(er_model.loglikelihood(adj_mat_preproc))
    assert(abs(er_model.loglikelihood(adj_mat_preproc) - our_er_loglik) < 1e-07)
    assert(abs(er_model.akaike(adj_mat_preproc) - (2 * (1 - our_er_loglik))) < 1e-07)
    print "akaike for ER graph: " + str(er_model.akaike(adj_mat_preproc))
    print "done testing basic object\n"


def test_learn_special_cases(adj_mat_infile):
    """
    Tests that Bernoulli model learns the right params and that Exponential model can be fit using learn_biment().

    """
    print "\n*** Testing param fitting for the bipartiteGraphModel class ***\n"
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    pi_vector_learned = score_data.learn_pi_vector(adj_mat)
    pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

    bernoulli_model = bipartite_fitting.learn_bernoulli(adj_mat_preproc)
    print "bernoulli model: checking pi_vector gets computed correctly"
    # make sure its pi_vector looks right
    assert (len(bernoulli_model.affil_params) == len(pi_vector_preproc))
    assert (max(abs(bernoulli_model.affil_params - pi_vector_preproc)) < 1e-07)

    # make sure its likelihood comes out the same as before
    # print "bernoulli model: checking akaike -- it's " + str(bernoulli_model.akaike(adj_mat_preproc))
    orig_bernoulli_model = bipartite_likelihood.bernoulliModel(pi_vector_preproc)
    assert(abs(bernoulli_model.akaike(adj_mat_preproc) - orig_bernoulli_model.akaike(adj_mat_preproc)) < 1e-07)
    # print "bernoulli log likelihood: " + str(bernoulli_model.loglikelihood(adj_mat_preproc))
    describe_exp_model(bernoulli_model, orig_bernoulli_model.loglikelihood(adj_mat_preproc), adj_mat_preproc)

    biment_model = bipartite_fitting.learn_biment(adj_mat_preproc)
    print "bipartite max entropy model: num_params is " + str(biment_model.get_num_params())
    describe_exp_model(biment_model, None, adj_mat_preproc)
    print "done testing special cases"


def test_learn_with_log_reg(adj_mat_infile):
    """
    Fits exponential models using the logistic regression mapping.
    Tries every possible combo of params to include and prints summary of learned params each time.
    Checks that the log likelihood of input graph matches sklearn's log likelihood of input data.

    """
    print "\n*** Testing param fitting for exponential models using logistic regression ***\n"
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    pi_vector_learned = score_data.learn_pi_vector(adj_mat)
    pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

    print "learning exponential model with only density param"
    model_only_dens, sklearn_ll = bipartite_fitting.learn_exponential_model(adj_mat_preproc, use_intercept=True,
                                              use_item_params = False, use_affil_params = False, withLL=True)
    describe_exp_model(model_only_dens, sklearn_ll, adj_mat_preproc)

    print "learning exponential model with affil params (should be like bernoulli)"
    model_only_affil, sklearn_ll = bipartite_fitting.learn_exponential_model(adj_mat_preproc, use_intercept=False,
                                              use_item_params = False, use_affil_params = True, withLL=True)
    describe_exp_model(model_only_affil, sklearn_ll, adj_mat_preproc)

    print "learning exponential model with density and affil params (does it beat bernoulli?)"
    model_dens_affil, sklearn_ll = bipartite_fitting.learn_exponential_model(adj_mat_preproc, use_intercept=True,
                                              use_item_params = False, use_affil_params = True, withLL=True)
    describe_exp_model(model_dens_affil, sklearn_ll, adj_mat_preproc)

    print "learning exponential model with item params"
    model_only_item, sklearn_ll = bipartite_fitting.learn_exponential_model(adj_mat_preproc, use_intercept=False,
                                              use_item_params = True, use_affil_params = False, withLL=True)
    describe_exp_model(model_only_item, sklearn_ll, adj_mat_preproc)

    print "learning exponential model with density and item params (does it beat items alone?)"
    model_dens_item, sklearn_ll = bipartite_fitting.learn_exponential_model(adj_mat_preproc, use_intercept=True,
                                              use_item_params = True, use_affil_params = False, withLL=True)
    describe_exp_model(model_dens_item, sklearn_ll, adj_mat_preproc)

    print "learning exponential model with item and affil params (is it like biment?)"
    model_item_affils, sklearn_ll = bipartite_fitting.learn_exponential_model(adj_mat_preproc, use_intercept=False,
                                              use_item_params = True, use_affil_params = True, withLL=True)
    describe_exp_model(model_item_affils, sklearn_ll, adj_mat_preproc)

    print "learning exponential model with density, item and affil params (is it the best?)"
    model_full, sklearn_ll = bipartite_fitting.learn_exponential_model(adj_mat_preproc, use_intercept=True,
                                              use_item_params = True, use_affil_params = True, withLL=True)
    describe_exp_model(model_full, sklearn_ll, adj_mat_preproc)
    print "finished!"
    print "done testing exponential model\n"


def describe_exp_model(model_obj, sklearn_ll, adj_mat):
    print "its likelihood: " + str(model_obj.loglikelihood(adj_mat))
    print "its akaike: " + str(model_obj.akaike(adj_mat))
    if sklearn_ll is not None:
        assert (abs(sklearn_ll - model_obj.loglikelihood(adj_mat)) < 1e-07)
    model_obj.print_params()


def demo_scoring_with_exp_model():
    """
    Uses exponential model within the new scoring method 'weighted_corr_exp'.

    """
    adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz"
    adj_mat = score_data.load_adj_mat(adj_mat_infile)

    print "version 1: make adj matrix dense"
    score_data.run_and_eval(adj_mat,
                            true_labels_func = score_data.true_labels_for_expts_with_5pairs,
                            method_spec=['weighted_corr', 'weighted_corr_exp'],
                            evals_outfile = "reality_appweek_50/python-out/exp_models-evals.txt",
                            pair_scores_outfile="reality_appweek_50/python-out/exp_models-scoredPairs.csv.gz",
                            print_timing=True)

    print "version 2: keep adj matrix sparse"
    score_data.run_and_eval(adj_mat,
                            true_labels_func = score_data.true_labels_for_expts_with_5pairs,
                            method_spec=['weighted_corr', 'weighted_corr_exp'],
                            evals_outfile = "reality_appweek_50/python-out/exp_models-evals-sparse.txt",
                            pair_scores_outfile="reality_appweek_50/python-out/exp_models-scoredPairs-sparse.csv.gz",
                            print_timing=True, make_dense=False)


def test_timing_for_fitting_model():
    # which way is faster, biment or logistic regression?
    print "\n*** Testing time to fit the exponential model ***\n"
    infile = "/Users/lfriedl/Documents/dissertation/real-data/brightkite/bipartite_adj.txt"

    num_nodes = (100, 200, 300, 400, 500)
    for num_to_try in num_nodes:
        print "\nnum_nodes = " + str(num_to_try)
        adj_mat, _ = loc_data.read_loc_adj_mat(infile, max_rows=num_to_try)
        pi_vector_learned = score_data.learn_pi_vector(adj_mat)
        pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

        start = timer()
        model1 = bipartite_fitting.learn_biment(adj_mat_preproc)
        end = timer()
        print "fit biment in " + str(end - start) + " seconds"
        describe_exp_model(model1, None, adj_mat_preproc)

        start = timer()
        model2, sklearn_ll = bipartite_fitting.learn_exponential_model(adj_mat_preproc, use_intercept=False,
                                                           use_item_params = True, use_affil_params = True)
        end = timer()
        print "fit with logistic regression in " + str(end - start) + " seconds"
        describe_exp_model(model2, sklearn_ll, adj_mat_preproc)

        # notes on computational complexity. both versions return 1 param per node (= num_items + num_affils).
        # biment: solver instantiates dense matrix the shape of adj_mat, memory-intensive. Could avoid that by
        # restoring original code, but all those entries need to be computed regardless.
        # logistic regression: number of entries of adj_mat becomes number of instances in the regression problem. The
        # sklearn call seems to use a lot of memory, even though it accepts a sparse input matrix.


# These tests aren't very rigorous, but they should all run w/o errors.
# (Going forward, tests.py makes sure that weighted_corr_exp scores doesn't change on the example data.)
if __name__ == "__main__":
    test_create_models()
    test_learn_special_cases(adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz")
    test_learn_with_log_reg(adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz")
    test_learn_special_cases(adj_mat_infile = "ng_aa_data2/data2_adjMat_quarterAffils.mtx.gz")
    test_learn_with_log_reg(adj_mat_infile = "ng_aa_data2/data2_adjMat_quarterAffils.mtx.gz")
    # demo_scoring_with_exp_model()
    # test_timing_for_fitting_model()