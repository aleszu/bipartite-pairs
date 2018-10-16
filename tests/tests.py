import sys
sys.path.append("../python-scoring")  # loads the path for now + later. But "import score_data" highlights as an error,
                                      # so pycharm-friendly substitute is below. (Need sys.path call to avoid errors
                                      # from imports within score_data.)
import imp
score_data = imp.load_source("score_data", "../python-scoring/score_data.py")
import numpy as np
import pandas as pd
import gzip
from timeit import default_timer as timer

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


# For now, just jaccard, but todo: others (all 12)
def test_pair_scores():
    print "Testing scores computed for pairs"
    # Starting from adj matrix, get phi (tested in test_adj_and_phi2 above), score pairs, and compare scores
    # to R's

    # Read adj data and prep pi_vector
    adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz"
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    pi_vector_learned = score_data.learn_pi_vector(adj_mat)
    pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

    methods_to_run = ['jaccard', 'cosine', 'cosineIDF', 'sharedSize', 'hamming', 'pearson', 'weighted_corr',
                      'shared_weight11', 'shared_weight1100', 'adamic_adar', 'newman', 'mixed_pairs']
    mixed_pairs_sims = [.01, .001]
    start = timer()
    # a test: is my 10x speedup simply from using a dense matrix?
    adj_mat_preproc = adj_mat_preproc.toarray()
    scores_data_frame = score_data.scoring_methods.score_pairs(score_data.gen_all_pairs, adj_mat_preproc,
                                                               which_methods=methods_to_run,
                                                               pi_vector=pi_vector_preproc, back_compat=True,
                                                               num_docs=adj_mat_preproc.shape[0],
                                                               mixed_pairs_sims=mixed_pairs_sims,
                                                               print_timing=True)
    scores_data_frame['label'] = score_data.get_true_labels_expt_data(score_data.gen_all_pairs(adj_mat), num_true_pairs=5)
    end = timer()
    print "ran " \
          + str(len(methods_to_run) + (len(mixed_pairs_sims) - 1 if 'mixed_pairs' in methods_to_run else 0)) \
          + " methods on " + str(adj_mat.shape[0] * (adj_mat.shape[0]-1)/float(2)) + " pairs"
    print "num seconds: " + str(end - start)

    # Read scores from R and compare
    mapping_from_R_methods = {"label":"label", "m":"sharedSize", "d": 'hamming', "one_over_log_p_m11": 'shared_weight11',
                              "one_over_log_p_m1100": 'shared_weight1100', "model5LogLR_t0.01": 'mixed_pairs_0.01',
                              "pearsonWeighted": 'weighted_corr',
                              "pearsonCorrZero": 'pearson', "adamicFixed": 'adamic_adar', "newmanCollab": 'newman',
                              "unweighted_cosineZero": 'cosine', "idfZero": 'cosineIDF', "jaccardSimZero": 'jaccard'}
    our_pi_methods = ['cosineIDF', 'weighted_corr', 'shared_weight11', 'shared_weight1100',
                      'adamic_adar', 'newman', 'mixed_pairs_0.01']

    scored_pairs_file_R = "reality_appweek_50/data50-inference-allto6.scoredPairs.csv.gz"
    with gzip.open(scored_pairs_file_R, 'r') as fpin:
        scores_data_frame_R = pd.read_csv(fpin)

    for (R_method, our_method) in mapping_from_R_methods.iteritems():
        if our_method in list(scores_data_frame):
            print "Checking " + our_method
            # R data doesn't have item numbers, but is in the same all-pairs order as ours
            print "max diff: " + str(abs(scores_data_frame[our_method] - scores_data_frame_R[R_method]).max() )

            # Sadly, the p_i vectors are off by a smidgen (see notes above), so anything that uses them can
            # differ too. sharedWeight11 vals differed by > 1e-06, and that was with only 65 affils.
            tolerance = 1e-03 if our_method in our_pi_methods else 1e-10  #
            assert(max(abs(scores_data_frame[our_method] - scores_data_frame_R[R_method])) < tolerance)

    return True


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

    wc_frame = score_data.scoring_methods.scoring_methods_fast.simple_only_weighted_corr(score_data.gen_all_pairs, adj_mat_preproc,
                                                                    pi_vector_preproc, print_timing=True)
    scored_pairs_file_R = "reality_appweek_50/data50-inference-allto6.scoredPairs.csv.gz"
    with gzip.open(scored_pairs_file_R, 'r') as fpin:
        scores_data_frame_R = pd.read_csv(fpin)

    print "Checking simple_only_weighted_corr"
    print "max diff: " + str(abs(wc_frame["weighted_corr"] - scores_data_frame_R["pearsonWeighted"]).max())
    assert (max(abs(wc_frame["weighted_corr"] - scores_data_frame_R["pearsonWeighted"])) < 1e-03)
    # Wow: it's 10 times faster than the usual method!
    # I tried implementing other methods the same way, and they were also faster
    # But eventually I figured out the savings was mainly because the sparse adj_matrix gets converted to dense,
    # which makes row access faster.




def test_eval_aucs():
    pass


if __name__ == "__main__":
    ok = test_adj_and_phi()
    test_adj_and_phi2()
    test_simple_jaccard()
    test_pair_scores()
    test_only_wc()
