import sys
sys.path.append("../python-scoring")  # loads the path for now + later. But "import score_data" highlights as an error,
                                      # so pycharm-friendly substitute is below. (Need sys.path call to avoid errors
                                      # from imports within score_data.)
sys.path.append("../expt-code")

import imp
score_data = imp.load_source("score_data", "../python-scoring/score_data.py")
scoring_methods = imp.load_source("scoring_methods", "../python-scoring/scoring_methods.py")
loc_data = imp.load_source("loc_data", "../expt-code/loc_data.py")
import numpy as np
import pandas as pd
import gzip
from timeit import default_timer as timer
import timeit
from sklearn.metrics import roc_auc_score, roc_curve, auc
from pympler import asizeof

mapping_from_R_methods = {"label": "label", "m": "shared_size", "d": 'hamming', "one_over_log_p_m11": 'shared_weight11',
                          "one_over_log_p_m1100": 'shared_weight1100', "model5LogLR_t0.01": 'mixed_pairs_0.01',
                          "pearsonWeighted": 'weighted_corr',
                          "pearsonCorrZero": 'pearson', "adamicFixed": 'adamic_adar', "newmanCollab": 'newman',
                          "unweighted_cosineZero": 'cosine', "idfZero": 'cosineIDF', "jaccardSimZero": 'jaccard'}
our_pi_methods = ['cosineIDF', 'weighted_corr', 'shared_weight11', 'shared_weight1100',
                  'adamic_adar', 'newman', 'mixed_pairs_0.01']


# Read the adj_mat, compute phi, and compare to
def test_adj_and_phi():
    print "Testing reading adjacency matrix and computing pi_vector"
    # Use the example data files "data15_*". They contain the contents of my expt data file alt.atheism/data1.Rdata

    #pi_vector_infile = "ng_aa_data1/data15_phi.txt.gz"  # this is from data1.Rdata, and it's the phi from the whole (larger) data set
    #pi_vector_whole_data = score_data.load_pi_from_file(pi_vector_infile) # ignoring this

    adj_mat_infile = "ng_aa_data1/data15_adj_mat.mtx.gz"

    # manually constructing these as I go along, using my existing R code in experimentRunner.R
    pi_vector_learned_R = score_data.load_pi_from_file("ng_aa_data1/data15.dataphi.txt.gz")
    pi_vector_preproc_R = score_data.load_pi_from_file("ng_aa_data1/data15.dataphipreproc.txt.gz")

    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    pi_vector_learned = score_data.learn_pi_vector(adj_mat)
    pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

    # Quirk from R: it saved floating point data with 7 digits of precision (see getOptions("digits") and format()).
    # Implication: if we want to ever use those phi files, should re-convert with higher precision.

    # For now, allow a difference of 1e-07 when comparing them

    # How annoying. Upping the precision simply revealed how I'm imprecise in the R code anyway. The Bernoulli <->
    # multinomial conversion I do doesn't keep the exact probabilities anyway. Actually... that's a possible bug. The
    # other time the code does this, it explicitly fixes that.

    # Compare. Expect pi_vector_learned to match pi_vector_learned_R and match numCols of adj_mat.
    assert(pi_vector_learned.shape[0] == adj_mat.shape[1])
    assert(max(abs(pi_vector_learned - pi_vector_learned_R)) < 1e-07)

    # Expect pi_vector_preproc to match pi_vector_preproc_R and match numCols of adj_mat_preproc
    assert(pi_vector_preproc.shape[0] == adj_mat_preproc.shape[1])
    assert(max(abs(pi_vector_preproc - pi_vector_preproc_R)) < 1e-07)

    # test flipping
    pi_vector_flipped_R = score_data.load_pi_from_file("ng_aa_data1/data15.dataphiflipped.txt.gz")
    adj_mat_flipped_R = score_data.load_adj_mat("ng_aa_data1/data15.adj_mat_flipped.mtx.gz")
    pi_vector_flipped, adj_mat_flipped = score_data.adjust_pi_vector(pi_vector_learned, adj_mat, flip_high_ps=True)
    # Expect the respective versions to match
    assert(pi_vector_flipped.shape == pi_vector_preproc.shape)
    assert(max(abs(pi_vector_flipped - pi_vector_flipped_R)) < 1e-07)
    assert(adj_mat_flipped_R.shape == adj_mat_flipped.shape)
    assert(abs(adj_mat_flipped_R - adj_mat_flipped).max() < 1e-07)

    return True

def test_adj_and_phi2():
    print "Testing reading adjacency matrix and computing pi_vector (2)"
    # Use something other than newsgroups! They're too complicated because they were run early.
    # Using "reality_appweek_50" subdir

    # Check that I can learn phi from the adjacency matrix and end up with the version in the inference file
    adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz"
    pi_vector_preproc_R = score_data.load_pi_from_file("reality_appweek_50/data50-inference-allto6.phi.csv.gz")

    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    pi_vector_learned = score_data.learn_pi_vector(adj_mat)
    pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

    # Expect pi_vector_preproc to match pi_vector_preproc_R
    assert(max(abs(pi_vector_preproc - pi_vector_preproc_R)) < 1e-07)

    return True


def test_pair_scores(adj_mat_infile, scored_pairs_file_R, make_dense=False, flip_high_ps=False):
    print "Testing scores computed for pairs"
    print "Adj matrix infile: " + adj_mat_infile + "; scored pairs file: " + scored_pairs_file_R
    # Starting from adj matrix, get phi (tested in test_adj_and_phi2 above), score pairs, and compare scores
    # to R's

    # Read adj data and prep pi_vector
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    pi_vector_learned = score_data.learn_pi_vector(adj_mat)
    pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat, flip_high_ps=flip_high_ps)

    methods_to_run = ['jaccard', 'cosine', 'cosineIDF', 'shared_size', 'hamming', 'pearson', 'weighted_corr',
                      'shared_weight11', 'shared_weight1100', 'adamic_adar', 'newman', 'mixed_pairs']
    mixed_pairs_sims = [.01, .001]
    start = timer()

    run_all = 2
    if make_dense:
        adj_mat_preproc = adj_mat_preproc.toarray()
    scores_data_frame = score_data.scoring_methods.score_pairs(score_data.gen_all_pairs, adj_mat_preproc,
                                                               which_methods=methods_to_run,
                                                               pi_vector=pi_vector_preproc, back_compat=True,
                                                               num_docs=adj_mat_preproc.shape[0],
                                                               mixed_pairs_sims=mixed_pairs_sims,
                                                               print_timing=True, run_all_implementations=run_all)
    scores_data_frame['label'] = score_data.get_true_labels_expt_data(score_data.gen_all_pairs(adj_mat), num_true_pairs=5)
    end = timer()
    print "ran " \
          + str(len(methods_to_run) + (len(mixed_pairs_sims) - 1 if 'mixed_pairs' in methods_to_run else 0)) \
          + " methods " + "(plus variants) " if run_all else "" \
          +  "on " + str(adj_mat.shape[0] * (adj_mat.shape[0]-1)/float(2)) + " pairs"
    print "num seconds: " + str(end - start)

    # Read scores from R and compare

    with gzip.open(scored_pairs_file_R, 'r') as fpin:
        scores_data_frame_R = pd.read_csv(fpin)

    for (R_method, our_method) in mapping_from_R_methods.iteritems():
        if our_method in list(scores_data_frame):
            print "Checking " + our_method
            # R data doesn't have item numbers, but is in the same all-pairs order as ours
            print "max diff: " + str(abs(scores_data_frame[our_method] - scores_data_frame_R[R_method]).max() )

            # Sadly, the p_i vectors are off by a smidgen (see notes above), so anything that uses them can
            # differ too. sharedWeight11 vals differed by > 1e-06, and that was with only 65 affils.
            tolerance = 1e-03 if our_method in our_pi_methods else 1e-10
            assert(max(abs(scores_data_frame[our_method] - scores_data_frame_R[R_method])) < tolerance)

    return scores_data_frame


def test_simple_jaccard():
    print "itty bitty jaccard test"
    item1 = np.zeros(shape=(5))
    item1[3] = item1[2] = 1
    item2 = np.array([1, 0, 1, 1, 0])
    print "jaccard gives: " + str(score_data.scoring_methods.jaccard(item1, item2))
    #print "and with extra ignored args, jaccard gives: " + str(score_data.scoring_methods.jaccard(item1, item2, extra=3, special=4))

def test_only_wc():
    # Read adj data and prep pi_vector
    adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz"
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    pi_vector_learned = score_data.learn_pi_vector(adj_mat)
    pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

    wc_frame = scoring_methods.extra_implementations.simple_only_weighted_corr(score_data.gen_all_pairs, adj_mat_preproc,
                                                                    pi_vector_preproc, print_timing=True)
    scored_pairs_file_R = "reality_appweek_50/data50-inference-allto6.scoredPairs.csv.gz"
    with gzip.open(scored_pairs_file_R, 'r') as fpin:
        scores_data_frame_R = pd.read_csv(fpin)

    print "Checking simple_only_weighted_corr"
    print "max diff: " + str(abs(wc_frame["weighted_corr"] - scores_data_frame_R["pearsonWeighted"]).max())
    assert (max(abs(wc_frame["weighted_corr"] - scores_data_frame_R["pearsonWeighted"])) < 1e-03)
    # Wow: it's 10 times faster than the usual method!
    # I tried implementing other methods the same way, and they were also faster
    # Eventually I figured out a big part of the savings was from the sparse adj_matrix getting converted to dense,
    # which makes row access faster.
    # But in fact, even after doing that, the initial-matrix-transformation way is still ~6x faster, wow.

    # Example timings for test_pair_scores() using sparse input matrix: (from when code converted it to dense)
    # weighted_corr: 0.272197961807 secs
    # simple_only_weighted_corr: 0.048122882843 secs
    # wc_transform: 0.051922082901 secs
    # ...and using dense input matrix:
    # weighted_corr: 0.042044878006 secs
    # simple_only_weighted_corr: 0.00608086585999 secs
    # wc_transform: 0.00586104393005 secs
    # One remaining uncertainty: is the difference still a fixed cost to convert to dense, or is it scaling differently?


# scores_and_labels: data frame created in test_pair_scores()
# aucs_file_R: 2-col text file, space separated. col 1: measure; col 2: value
def test_eval_aucs(scores_and_labels, aucs_file_R):
    print "Checking AUCs against " + aucs_file_R

    with open(aucs_file_R, 'r') as fpin:
        for line in fpin:
            measure, value = line.split()
            if measure[:4] != "auc_":
                continue

            R_method = measure[4:]
            our_method = mapping_from_R_methods.get(R_method, None)
            if (our_method is not None) and (our_method in list(scores_and_labels)):

                # special case: the old hamming score (being tested here) needed its direction to be flipped.
                if our_method == "hamming":
                    scores_and_labels[our_method] = -1 * scores_and_labels[our_method]

                our_auc = roc_auc_score(y_true=scores_and_labels['label'], y_score=scores_and_labels[our_method])

                # oddly, python's auc computation comes out a bit different than R's for cosine and pearson
                # (on reality example), even though the indiv scores match up to 1e-15. Why?

                # (my changing this one arg doesn't fix it)
                #my_redone_auc = auc_all_pts(scores_and_labels[our_method], scores_and_labels['label'])

                #  --> Seems to be a question of floating point precision: python decides some scores differ at the
                # > 16th digit, when R thinks they're identical. (Spot checking: they should be identical.)
                # There's a discussion on the scipy github about whether to do this or not -- they used to allow some
                # tolerance, but then it made mistakes the other direction.

                R_auc = float(value)
                auc_diff = abs(our_auc - R_auc)
                #auc_diff = abs(my_redone_auc - R_auc)

                print "AUC diff for " + our_method + ": " + str(auc_diff)

                # large tolerance because the pearson example differs by .0004.
                # Interestingly, fewer affils --> more ties --> more difference. The newsgroups example would work
                # with tolerance = 1e-13.
                tolerance = 1e-03
                assert(abs(our_auc - R_auc) < tolerance)


def auc_all_pts(scores, labels):
    y_true = [int(x) for x in labels]  # change T/F to 1/0
    fpr, tpr, _ = roc_curve(y_true, scores, drop_intermediate=False)
    return auc(fpr, tpr)


def resources_test(run_all_implementations=True):
    # Let's read in portions of a big matrix in increasing size, and for each size, score all pairs (both sparse and dense).
    # This will let us see how things scale and where memory limits will come in.
    infile = "/Users/lfriedl/Documents/dissertation/real-data/brightkite/bipartite_adj.txt"

    num_nodes = (100, 1000, 10000, 100000)
    num_nodes = [10000] # this size: no run finished in the length of time I was willing to wait
    num_nodes = [500]
    for num_to_try in num_nodes:
        adj_mat, _ = loc_data.read_loc_adj_mat(infile, max_rows=num_to_try)

        pi_vector_learned = score_data.learn_pi_vector(adj_mat)
        pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

        # (order given here doesn't matter)
        methods_to_run = ['cosine', 'cosineIDF',
                          # use fast "transform"
                          'shared_size', 'adamic_adar', 'newman', 'shared_weight11',
                          # medium
                          'hamming', 'pearson',  'jaccard',
                          # WC uses "transform" when dense, "terms" when sparse -- speed varies accordingly
                          'weighted_corr',
                          # only have slow "terms" method
                           'shared_weight1100', 'mixed_pairs']

        adj_mat_preproc_dense = adj_mat_preproc.toarray()
        print "\ndense version takes up " + str(sys.getsizeof(adj_mat_preproc_dense)) + " bytes"

        start = timer()
        scores_data_frame = score_data.scoring_methods.score_pairs(score_data.gen_all_pairs, adj_mat_preproc_dense,
                                                                   which_methods=methods_to_run,
                                                                   pi_vector=pi_vector_preproc, back_compat=True,
                                                                   num_docs=adj_mat_preproc.shape[0],
                                                                   mixed_pairs_sims=[.01],
                                                                   print_timing=True,
                                                                   run_all_implementations=run_all_implementations)
        end = timer()
        print "for matrix with " + str(adj_mat_preproc.shape[0]) + " items, " + str(adj_mat_preproc.shape[1]) \
            + " affils, "
        print "ran all methods using dense matrix in " + str(end - start) + " seconds"

        print "\nsparse adj_matrix takes up " + str(asizeof.asizeof(adj_mat_preproc)) + " bytes;"

        start = timer()
        scores_data_frame = score_data.scoring_methods.score_pairs(score_data.gen_all_pairs, adj_mat_preproc,
                                                                   which_methods=methods_to_run,
                                                                   pi_vector=pi_vector_preproc, back_compat=True,
                                                                   num_docs=adj_mat_preproc.shape[0],
                                                                   mixed_pairs_sims=[.01],
                                                                   print_timing=True,
                                                                   run_all_implementations=run_all_implementations)
        end = timer()
        print "for matrix with " + str(adj_mat_preproc.shape[0]) + " items, " + str(adj_mat_preproc.shape[1]) \
            + " affils, "
        print "ran all methods using sparse matrix in " + str(end - start) + " seconds"

# To nail down which versions of a few methods are fastest
def test_timings(infile):

    print "testing timings using infile " + infile
    setup = """\
import imp
score_data = imp.load_source("score_data", "../python-scoring/score_data.py")
scoring_methods = imp.load_source("score_data", "../python-scoring/scoring_methods.py")
""" + \
"adj_mat_infile = \"" + infile + """\" 
adj_mat = score_data.load_adj_mat(adj_mat_infile)
pi_vector_learned = score_data.learn_pi_vector(adj_mat)
pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)
"""

    s = """\
    scoring_methods.scoring_methods_fast.simple_only_weighted_corr(score_data.gen_all_pairs, adj_mat_preproc,
                                                                    pi_vector_preproc, print_timing=False)
    """
    print "simple_only_weighted_corr:"
    print timeit.timeit(s, setup=setup, number=100)

    s2 = """

scoring_methods.compute_scores_with_transform(score_data.gen_all_pairs, adj_mat_preproc, 
                                scoring_methods.wc_transform, pi_vector=pi_vector_preproc, print_timing=False)
    """
    print "wc_transform:"
    print timeit.timeit(s2, setup=setup, number=100)
    # --> wc_transform is faster for weighted_corr (both made matrix dense)

    s3 = """
scoring_methods.compute_scores_from_terms(score_data.gen_all_pairs, adj_mat_preproc, scoring_methods.wc_terms,
                        pi_vector=pi_vector_preproc,
                        num_affils=adj_mat_preproc.shape[1], print_timing=False)
    """
    print "wc_terms:"
    print timeit.timeit(s3, setup=setup, number=100)
    s4 = """
 scoring_methods.scoring_methods_fast.simple_weighted_corr_sparse(score_data.gen_all_pairs, adj_mat_preproc,
                                               pi_vector=pi_vector_preproc,
                                               print_timing=False)    
    """
    print "simple_weighted_corr_sparse:"
    print timeit.timeit(s4, setup=setup, number=100)
    # --> for sparse matrix, wc_terms is faster than simple_weighted_corr_sparse


    # Adamic-adar
    s5 = """
scoring_methods.compute_scores_with_transform(score_data.gen_all_pairs, adj_mat_preproc,
                                              scoring_methods.adamic_adar_transform, 
                                                      num_docs=adj_mat_preproc.shape[0],
                                                      pi_vector=pi_vector_preproc)
    """
    print "adamic_adar_transform:"
    print timeit.timeit(s5, setup=setup, number=100)

    s6 = """
num_docs_word_occurs_in = np.maximum(adj_mat_preproc.shape[0] * pi_vector_preproc, 2)
scoring_methods.extra_implementations.simple_only_adamic_adar_scores(score_data.gen_all_pairs, adj_mat_preproc,
                                                                                     num_docs_word_occurs_in)
"""
    print "simple_only_adamic_adar_scores:"
    print timeit.timeit(s5, setup=setup, number=100)
    # --> simple_only_adamic_adar_scores sometimes faster, sometimes slower

    s6 = "scoring_methods.scoring_methods_fast.simple_only_phi_coeff(score_data.gen_all_pairs, adj_mat_preproc)"
    print "simple_only_phi_coeff:"
    print timeit.timeit(s6, setup=setup, number=100)

    s7 = "scoring_methods.extra_implementations.simple_only_pearson(score_data.gen_all_pairs, adj_mat_preproc)"
    print "simple_only_pearson:"
    print timeit.timeit(s7, setup=setup, number=100)
    # simple_only_pearson was a bit faster



# Just tests that it runs; outfiles can be manually compared with backed-up copies if desired
def test_run_and_eval():
    # infile --> adj matrix (now moved outside the function)
    adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz"
    adj_mat = score_data.load_adj_mat(adj_mat_infile)

    score_data.run_and_eval(adj_mat,
                            true_labels_func = score_data.true_labels_for_expts_with_5pairs,
                            method_spec="all",
                            evals_outfile = "reality_appweek_50/python-out/evals-basic.txt",
                            pair_scores_outfile="reality_appweek_50/python-out/scoredPairs-basic.csv.gz")


def test_loc_data():
    # todo: set random seed so this is actually repeatable
    adj_mat_infile = '/Users/lfriedl/Documents/dissertation/real-data/brightkite/bipartite_adj.txt'
    edges_infile = '/Users/lfriedl/Documents/dissertation/real-data/brightkite/loc-brightkite_edges.txt'
    rows_outfile = 'brightkite/data-ex1.txt'
    adj_mat, row_labels, label_generator = loc_data.read_sample_save(adj_mat_infile, edges_infile, num_nodes=300, rows_outfile=rows_outfile)
    if label_generator is None:
        print "Found no edges; stopping"

    else:
        score_data.run_and_eval(adj_mat, true_labels_func = label_generator, method_spec="all",
                                evals_outfile = "brightkite/evals-ex1.txt",
                                pair_scores_outfile="brightkite/scoredPairs-ex1.csv.gz", row_labels=row_labels,
                                print_timing=True)


if __name__ == "__main1__":
    resources_test(run_all_implementations=False)
    test_timings("ng_aa_data2/data2_adjMat_quarterAffils.mtx.gz")
    test_timings("reality_appweek_50/data50_adjMat.mtx.gz")
    test_loc_data()



if __name__ == "__main__":
    ok = test_adj_and_phi()
    test_adj_and_phi2()
    #test_simple_jaccard()
    test_only_wc()

    # Test reality mining example

    # note: run_all_implementations flag inside score_pairs() means "run and time all versions",
    # but we only look at the output of the last
    print "Reality mining, data set #50 -- as sparse matrix"
    test_pair_scores(adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz",
                     scored_pairs_file_R = "reality_appweek_50/data50-inference-allto6.scoredPairs.csv.gz")
    print "Reality mining, data set #50 -- as dense matrix"
    scores_frame = test_pair_scores(adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz",
                     scored_pairs_file_R = "reality_appweek_50/data50-inference-allto6.scoredPairs.csv.gz",
                     make_dense=True)  # much faster. But won't scale to large matrices.
    # Test AUCs for reality ex
    test_eval_aucs(scores_frame, aucs_file_R = "reality_appweek_50/results50.txt")


    # Test newsgroups example (plain run was too complicated, but flipped run was later, so
    # more standardized)
    print "Newsgroups, data set #2, flipped -- as sparse matrix"
    test_pair_scores(adj_mat_infile = "ng_aa_data2/data2_adjMat_quarterAffils.mtx.gz",
                     scored_pairs_file_R = "ng_aa_data2/data2-inferenceFlip.scoredPairs.csv.gz",
                     flip_high_ps=True)
    print "Newsgroups, data set #2, flipped -- as dense matrix"
    scores_frame = test_pair_scores(adj_mat_infile = "ng_aa_data2/data2_adjMat_quarterAffils.mtx.gz",
                     scored_pairs_file_R = "ng_aa_data2/data2-inferenceFlip.scoredPairs.csv.gz",
                     flip_high_ps=True, make_dense=True)

    # Test AUCs for newsgroups ex
    test_eval_aucs(scores_frame, aucs_file_R = "ng_aa_data2/results2-flip_allto6.txt")

    # The function that does it all, that we'll usually call
    test_run_and_eval()
