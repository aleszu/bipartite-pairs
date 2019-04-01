from __future__ import print_function
from builtins import zip, str
import sys
sys.path.append("../python-scoring")  # add other dirs to path (for non-PyCharm use)
sys.path.append("../expt-code")

import score_data       # (got rid of clunky imp.load_source calls by adding other dirs to path in PyCharm Prefs)
import scoring_methods
import loc_data
import pandas as pd
import gzip
from timeit import default_timer as timer
from sklearn.metrics import roc_auc_score, roc_curve, auc


mapping_from_R_methods = {"label": "label", "m": "shared_size", "d": 'hamming', "one_over_log_p_m11": 'shared_weight11',
                          "one_over_log_p_m1100": 'shared_weight1100', "model5LogLR_t0.01": 'mixed_pairs_0.01',
                          "pearsonWeighted": 'weighted_corr',
                          "pearsonCorrZero": 'pearson', "adamicFixed": 'adamic_adar', "newmanCollab": 'newman',
                          "unweighted_cosineZero": 'cosine', "idfZero": 'cosineIDF', "jaccardSimZero": 'jaccard'}
our_pi_methods = ['cosineIDF', 'weighted_corr', 'shared_weight11', 'shared_weight1100',
                  'adamic_adar', 'newman', 'mixed_pairs_0.01']


# tolerances: using handful of example data sets, chosen to work around some weird quirks; see comments.
# Hard-coded as 1e-07 for pi_vector, 1e-10 for an indiv score (1e-05 if it uses [R's iffy] pi_vector, and sadly 1e-02
# when comparing to FAISS-produced scores), and 1e-03 for AUCs.


def test_adj_and_phi():
    """
    Reads adj matrix, makes sure we can match what R code did for learning pi_vector, preprocessing it, and flipping it.

    Uses & compares to files: 'ng_aa_data1/data15' . [_adj_mat.mtx.gz, .dataphi.txt.gz, .dataphipreproc.txt.gz,
                                                      .dataphiflipped.txt.gz, .adj_mat_flipped.mtx.gz]

    Throws assertion error if unhappy
    """
    print("\n*** Testing reading adjacency matrix and computing pi_vector ***\n")
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


def test_adj_and_phi2():
    """
    Reads adj matrix, checks that we can learn pi_vector for a second data set.
    Using files: "reality_appweek_50/data50_adjMat.mtx.gz", "reality_appweek_50/data50-inference-allto6.phi.csv.gz"
    """
    print("\n*** Testing reading adjacency matrix and computing pi_vector (2) ***\n")
    # Use something other than newsgroups! They're too complicated because they were run early.

    # Check that I can learn phi from the adjacency matrix and end up with the version in the inference file
    adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz"
    pi_vector_preproc_R = score_data.load_pi_from_file("reality_appweek_50/data50-inference-allto6.phi.csv.gz")

    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    pi_vector_learned = score_data.learn_pi_vector(adj_mat)
    pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

    # Expect pi_vector_preproc to match pi_vector_preproc_R
    assert(max(abs(pi_vector_preproc - pi_vector_preproc_R)) < 1e-07)


def test_pair_scores_against_R(adj_mat_infile, scored_pairs_file_R, scored_pairs_file_new, make_dense=False,
                               flip_high_ps=False, run_all=0, prefer_faiss=False):
    """
    Starting from an adj matrix, score pairs (using current implementation) and compare to reference file run from R.
    Similar contents to score_data.run_and_eval().

    :param run_all: set to 2 (or 1) to run and time all (or more) implementations.
                    However, we only look at the scores of the last one.
    """
    print("\n*** Testing scores computed for pairs ***\n")
    print("Adj matrix infile: " + adj_mat_infile + "; scored pairs reference file: " + scored_pairs_file_R)

    # Read adj data and prep pi_vector
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    pi_vector_learned = score_data.learn_pi_vector(adj_mat)
    pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat, flip_high_ps=flip_high_ps)

    methods_to_run = ['jaccard', 'cosine', 'cosineIDF', 'shared_size', 'hamming', 'pearson', 'weighted_corr',
                      'shared_weight11', 'shared_weight1100', 'adamic_adar', 'newman', 'mixed_pairs']
    mixed_pairs_sims = [.01, .001]
    start = timer()

    if make_dense:
        adj_mat_preproc = adj_mat_preproc.toarray()
    scoring_methods.score_pairs(score_data.gen_all_pairs, adj_mat_preproc, which_methods=methods_to_run,
                                outfile_csv_gz=scored_pairs_file_new, pi_vector=pi_vector_preproc, back_compat=True,
                                num_docs=adj_mat_preproc.shape[0], mixed_pairs_sims=mixed_pairs_sims,
                                print_timing=True, run_all_implementations=run_all, prefer_faiss=prefer_faiss)
    with gzip.open(scored_pairs_file_new, 'r') as fpin:
        scores_data_frame = pd.read_csv(fpin)

    scores_data_frame['label'] = score_data.get_true_labels_expt_data(score_data.gen_all_pairs(adj_mat), num_true_pairs=5)
    end = timer()
    print("ran " \
          + str(len(methods_to_run) + (len(mixed_pairs_sims) - 1 if 'mixed_pairs' in methods_to_run else 0)) \
          + " methods " + ("(plus variants) " if run_all else "") \
          +  "on " + str(adj_mat.shape[0] * (adj_mat.shape[0]-1)/float(2)) + " pairs")
    print("num seconds: " + str(end - start))

    # Read scores from R and compare
    with gzip.open(scored_pairs_file_R, 'r') as fpin:
        scores_data_frame_R = pd.read_csv(fpin)

    for (R_method, our_method) in mapping_from_R_methods.items():
        if our_method in list(scores_data_frame):
            print("Checking " + our_method)
            # R data doesn't have item numbers, but is in the same all-pairs order as ours
            print("max diff: " + str(abs(scores_data_frame[our_method] - scores_data_frame_R[R_method]).max() ))

            # Sadly, the p_i vectors are off by a smidgen (see notes above), so anything that uses them can
            # differ too. sharedWeight11 vals differed by > 1e-06, and that was with only 65 affils.
            tolerance = 1e-10
            if prefer_faiss:
                tolerance = 1e-04
            elif our_method in our_pi_methods:
                tolerance = 1e-05
            assert(max(abs(scores_data_frame[our_method] - scores_data_frame_R[R_method])) < tolerance)

    return scores_data_frame


# scores_and_labels: data frame created in test_pair_scores()
# aucs_file_R: 2-col text file, space separated. col 1: measure; col 2: value
def test_eval_aucs(scores_and_labels, aucs_file_R, tolerance = 1e-07):
    print("\n*** Checking AUCs against " + aucs_file_R + " ***\n")

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

                R_auc = float(value)
                auc_diff = abs(our_auc - R_auc)
                print("AUC diff for " + our_method + ": " + str(auc_diff))
                assert(auc_diff < tolerance)

                # oddly, python's auc computation comes out a bit different than R's for cosine and pearson
                # (on reality example), even though the indiv scores match up to 1e-15.

                #  --> Seems to be a question of floating point precision: python decides some scores differ at the
                # > 16th digit, when R thinks they're identical. (Spot checking: they should be identical.)
                # There's a discussion on the scipy github about whether to do this or not -- they used to allow some
                # tolerance, but then it made mistakes the other direction.
                # (e.g., https://github.com/scikit-learn/scikit-learn/issues/3864)

                # Interestingly, fewer affils --> more tie scores --> more difference. The newsgroups example would work
                # with tolerance = 1e-13, but in reality example, pearson differs by .0004.

                # (changing the arg 'drop_intermediate' had no effect)
                #my_redone_auc = auc_all_pts(scores_and_labels[our_method], scores_and_labels['label'])

# def auc_all_pts(scores, labels):
#     y_true = [int(x) for x in labels]  # change T/F to 1/0
#     fpr, tpr, _ = roc_curve(y_true, scores, drop_intermediate=False)
#     return auc(fpr, tpr)



def test_only_wc(adj_mat_infile, scored_pairs_file_R):
    """
    Like test_pair_scores_against_R(), but checks scores & timing of the function simple_only_weighted_corr().
    (This was the first scoring method I implemented using a transform of the adj_matrix.)

    :param adj_mat_infile: local path ending in .mtx.gz
    :param scored_pairs_file_R: local path ending in .csv.gz
    """

    print("\n*** Checking simple_only_weighted_corr against scores from R ***\n")

    # Read adj data and prep pi_vector
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    pi_vector_learned = score_data.learn_pi_vector(adj_mat)
    pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

    wc_frame = scoring_methods.extra_implementations.simple_only_weighted_corr(score_data.gen_all_pairs, adj_mat_preproc,
                                                                    pi_vector_preproc, print_timing=True)
    with gzip.open(scored_pairs_file_R, 'r') as fpin:
        scores_data_frame_R = pd.read_csv(fpin)

    print("max diff: " + str(abs(wc_frame["weighted_corr"] - scores_data_frame_R["pearsonWeighted"]).max()))
    assert (max(abs(wc_frame["weighted_corr"] - scores_data_frame_R["pearsonWeighted"])) < 1e-05)
    # Wow: it's 10 times faster than the usual method!
    # I tried implementing other methods the same way, and they were also faster
    # One part of the savings was from the sparse adj_matrix getting converted to dense, which makes row access faster.
    # But even after doing that, the initial-matrix-transformation way is still ~6x faster, wow.

    # Example timings for test_pair_scores() using sparse input matrix: (from when code converted it to dense)
    # weighted_corr: 0.272197961807 secs
    # simple_only_weighted_corr: 0.048122882843 secs
    # wc_transform: 0.051922082901 secs
    # ...and using dense input matrix:
    # weighted_corr: 0.042044878006 secs
    # simple_only_weighted_corr: 0.00608086585999 secs
    # wc_transform: 0.00586104393005 secs
    # One remaining uncertainty: is the difference still a fixed cost to convert to dense, or is it scaling differently?


def test_all_methods_no_changes(adj_mat_infile, results_dir, prefer_faiss=False):
    print("\n*** Checking whether all scores match this package's previous version, for " + results_dir + " ***\n")

    orig_pair_scores_file = results_dir + "/scoredPairs-basic.csv.gz.bak"
    orig_evals_file = results_dir + "/evals-basic.txt.bak"

    new_pair_scores_file = results_dir + "/scoredPairs-basic.csv.gz"
    new_evals_file = results_dir + "/evals-basic.txt"

    # Run all methods the usual way on reality_appweek_50
    demo_run_and_eval(adj_mat_infile=adj_mat_infile,
                      pair_scores_outfile=new_pair_scores_file, evals_outfile=new_evals_file,
                      prefer_faiss=prefer_faiss)

    # compare pair scores to stored version (gzip header of files will differ)
    print("Checking pair scores")
    with gzip.open(orig_pair_scores_file, 'r') as fpin:
        orig_scores_data_frame = pd.read_csv(fpin)
    with gzip.open(new_pair_scores_file, 'r') as fpin:
        new_scores_data_frame = pd.read_csv(fpin)
    # assert(new_scores_data_frame.equals(orig_scores_data_frame))  # may need to compare using a tolerance later
    tolerance = 2 if prefer_faiss else 10
    pd.testing.assert_frame_equal(orig_scores_data_frame, new_scores_data_frame, check_exact=False,
                                  check_less_precise=tolerance)

    # compare evals to stored version. (Simpler way to compare contents of two files.)
    print("Checking AUCs/evals files")
    with open(orig_evals_file, 'r') as f1, open(new_evals_file, 'r') as f2:
        for line1, line2 in zip(f1, f2):
            measure1, value1 = line1.split()
            measure2, value2 = line2.split()
            assert(measure1 == measure2)
            assert(abs(float(value1) - float(value2)) < 1e-03)



# (Renamed to "demo" because it's not testing anything, just runs.)
def demo_run_and_eval(adj_mat_infile, pair_scores_outfile, evals_outfile, prefer_faiss=False):

    adj_mat = score_data.load_adj_mat(adj_mat_infile)

    score_data.run_and_eval(adj_mat,
                            true_labels_func=score_data.true_labels_for_expts_with_5pairs,
                            method_spec="all",
                            evals_outfile=evals_outfile,
                            pair_scores_outfile=pair_scores_outfile,
                            print_timing=True, prefer_faiss=prefer_faiss)


def demo_loc_data():
    # todo: set random seed so this is actually repeatable
    adj_mat_infile = '/Users/lfriedl/Documents/dissertation/real-data/brightkite/bipartite_adj.txt'
    edges_infile = '/Users/lfriedl/Documents/dissertation/real-data/brightkite/loc-brightkite_edges.txt'
    rows_outfile = 'brightkite/data-ex1.txt'
    adj_mat, row_labels, label_generator = loc_data.read_sample_save(adj_mat_infile, edges_infile, num_nodes=300, rows_outfile=rows_outfile)
    if label_generator is None:
        print("Found no edges; stopping")

    else:
        score_data.run_and_eval(adj_mat, true_labels_func = label_generator, method_spec="all",
                                evals_outfile = "brightkite/evals-ex1.txt",
                                pair_scores_outfile="brightkite/scoredPairs-ex1.csv.gz", row_labels=row_labels,
                                print_timing=True)

# useful for one-offs
if __name__ == "__main0__":
    score_data.score_only('../../CHASE-expts/1_42wc-1-1smerged-plots/1312970400_graph.mtx.gz', ['weighted_corr_exp'],
                          '../../CHASE-expts/1_42wc-1-1smerged-plots/1312970400_new_pair_scores.csv.gz', print_timing=True)
    # demo_loc_data()

    # The function that does it all, that we'll usually call
    # demo_run_and_eval(adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz",
    #                   pair_scores_outfile="reality_appweek_50/python-out/scoredPairs-basic.csv.gz",
    #                   evals_outfile = "reality_appweek_50/python-out/evals-basic.txt")
    # test_all_methods_no_changes(adj_mat_infile="reality_appweek_50/data50_adjMat.mtx.gz",
    #                             results_dir="reality_appweek_50/python-out")


# Everything in "main" actually tests things and should run w/o errors.
if __name__ == "__main__":
    test_adj_and_phi()
    test_adj_and_phi2()

    tmp_scored_pairs_file_new = 'reality_appweek_50/tmp.scoredPairs.csv.gz'
    # Test a specific implementation of weighted_corr
    test_only_wc(adj_mat_infile ="reality_appweek_50/data50_adjMat.mtx.gz",
                 scored_pairs_file_R = "reality_appweek_50/data50-inference-allto6.scoredPairs.csv.gz")

    # Test reality mining example
    print("\nReality mining, data set #50 -- as sparse matrix")
    test_pair_scores_against_R(adj_mat_infile ="reality_appweek_50/data50_adjMat.mtx.gz",
                               scored_pairs_file_R = "reality_appweek_50/data50-inference-allto6.scoredPairs.csv.gz",
                               scored_pairs_file_new = tmp_scored_pairs_file_new)
    print("\nReality mining, data set #50 -- as dense matrix")
    scores_frame = test_pair_scores_against_R(adj_mat_infile ="reality_appweek_50/data50_adjMat.mtx.gz",
                                              scored_pairs_file_R = "reality_appweek_50/data50-inference-allto6.scoredPairs.csv.gz",
                                              scored_pairs_file_new=tmp_scored_pairs_file_new,
                                              make_dense=True)  # much faster. But won't scale to large matrices.
    print("\nReality mining, data set #50 -- dense with FAISS")
    scores_frame = test_pair_scores_against_R(adj_mat_infile ="reality_appweek_50/data50_adjMat.mtx.gz",
                                              scored_pairs_file_R = "reality_appweek_50/data50-inference-allto6.scoredPairs.csv.gz",
                                              scored_pairs_file_new=tmp_scored_pairs_file_new,
                                              make_dense=True, prefer_faiss=True)
    # Test AUCs for reality ex
    test_eval_aucs(scores_frame, aucs_file_R = "reality_appweek_50/results50.txt", tolerance = 1e-03)


    # Test newsgroups example (plain run was too complicated to replicate, but flipped run was later, so
    # more standardized)
    tmp_scored_pairs_file_new = 'ng_aa_data2/tmp.scoredPairs.csv.gz'
    print("\nNewsgroups, data set #2, flipped -- as sparse matrix")
    test_pair_scores_against_R(adj_mat_infile ="ng_aa_data2/data2_adjMat_quarterAffils.mtx.gz",
                               scored_pairs_file_R = "ng_aa_data2/data2-inferenceFlip.scoredPairs.csv.gz",
                               scored_pairs_file_new=tmp_scored_pairs_file_new,
                               flip_high_ps=True)
    print("\nNewsgroups, data set #2, flipped -- as dense matrix")
    scores_frame = test_pair_scores_against_R(adj_mat_infile ="ng_aa_data2/data2_adjMat_quarterAffils.mtx.gz",
                                              scored_pairs_file_R = "ng_aa_data2/data2-inferenceFlip.scoredPairs.csv.gz",
                                              scored_pairs_file_new=tmp_scored_pairs_file_new,
                                              flip_high_ps=True, make_dense=True)
    print("\nNewsgroups, data set #2, flipped -- dense with FAISS")
    scores_frame = test_pair_scores_against_R(adj_mat_infile ="ng_aa_data2/data2_adjMat_quarterAffils.mtx.gz",
                                              scored_pairs_file_R = "ng_aa_data2/data2-inferenceFlip.scoredPairs.csv.gz",
                                              scored_pairs_file_new=tmp_scored_pairs_file_new,
                                              flip_high_ps=True, make_dense=True, prefer_faiss=True)
    # Test AUCs for newsgroups ex
    test_eval_aucs(scores_frame, aucs_file_R = "ng_aa_data2/results2-flip_allto6.txt", tolerance = 1e-04)

    # Test all scoring methods against what this package produced earlier
    print("\n*** Running and saving aucs file to compare with our standard one ***\n")
    test_all_methods_no_changes(adj_mat_infile ="reality_appweek_50/data50_adjMat.mtx.gz",
                                results_dir="reality_appweek_50/python-out")

    print("\n*** Running with FAISS and saving aucs file to compare with our standard one ***\n")
    test_all_methods_no_changes(adj_mat_infile="reality_appweek_50/data50_adjMat.mtx.gz",
                                results_dir="reality_appweek_50/python-out", prefer_faiss=True)
    # todo: set up this call for additional data sets
