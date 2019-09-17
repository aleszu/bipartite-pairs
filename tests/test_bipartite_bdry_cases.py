
import numpy as np
import sys
sys.path.append("../python-scoring")  # add other dirs to path (for non-PyCharm use)
sys.path.append("../expt-code")

import score_data
import expts_labeled_data


# Note: in original adj_mat for reality_appday_94, item[26,] is all 0's except for at affil[,115], which is all 1's.

# Cases 1-3 use default remove_boundary_items=True
def case1_no_bdry_nodes(adj_mat_infile, results_dir, aucs_file_to_match):
    print("\nCase 1\n")
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    new_evals_file = results_dir + "/evals-case1.txt"
    score_data.run_and_eval(adj_mat,
                            true_labels_func=expts_labeled_data.true_labels_for_expts_with_5pairs,
                            method_spec=['weighted_corr', 'weighted_corr_exp'],
                            evals_outfile=new_evals_file,
                            pair_scores_outfile=None,
                            print_timing=True)
    compare_auc_files(new_evals_file, aucs_file_to_match)


def case2_keep_0affils(adj_mat_infile, results_dir, aucs_file_to_match):
    print("\nCase 2\n")
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    # Want item 26 to stay out, and also want only all-0 affils, so set affil[,115] to all 0's
    adj_mat[:,115] = 0
    new_evals_file = results_dir + "/evals-case2.txt"
    score_data.run_and_eval(adj_mat,
                            true_labels_func=expts_labeled_data.true_labels_for_expts_with_5pairs,
                            method_spec=['weighted_corr', 'weighted_corr_exp'],
                            evals_outfile=new_evals_file,
                            pair_scores_outfile=None,
                            print_timing=True, remove_boundary_affils=False)
    compare_auc_files(new_evals_file, aucs_file_to_match)


def case3_keep_0and1affils(adj_mat_infile, results_dir, aucs_file_to_match):
    print("\nCase 3\n")
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    # Want the natural all-0 and all-1 affils, but still want item 26 to stay out.
    adj_mat[26,:] = 0
    new_evals_file = results_dir + "/evals-case3.txt"
    score_data.run_and_eval(adj_mat,
                            true_labels_func=expts_labeled_data.true_labels_for_expts_with_5pairs,
                            method_spec=['weighted_corr', 'weighted_corr_exp'],
                            evals_outfile=new_evals_file,
                            pair_scores_outfile=None,
                            print_timing=True, remove_boundary_affils=False)
    compare_auc_files(new_evals_file, aucs_file_to_match)


# Cases 4-6: same as 1-3, except remove_boundary_items=False, so we keep the all-0 item
def case4_0item_no_bdry_affils(adj_mat_infile, results_dir, aucs_file_to_match):
    print("\nCase 4\n")
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    # Keep the natural all-0 item, and tell program to remove boundary affils
    new_evals_file = results_dir + "/evals-case4.txt"
    score_data.run_and_eval(adj_mat,
                            true_labels_func=expts_labeled_data.true_labels_for_expts_with_5pairs,
                            method_spec=['weighted_corr', 'weighted_corr_exp'],
                            evals_outfile=new_evals_file,
                            pair_scores_outfile=None,
                            print_timing=True, remove_boundary_items=False, remove_boundary_affils=True)
    compare_auc_files(new_evals_file, aucs_file_to_match)


def case5_0item_keep_0affils(adj_mat_infile, results_dir, aucs_file_to_match):
    print("\nCase 5\n")
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    # want only all-0 affils, so set affil[,115] to all 0's
    adj_mat[:, 115] = 0
    new_evals_file = results_dir + "/evals-case5.txt"
    score_data.run_and_eval(adj_mat,
                            true_labels_func=expts_labeled_data.true_labels_for_expts_with_5pairs,
                            method_spec=['weighted_corr', 'weighted_corr_exp'],
                            evals_outfile=new_evals_file,
                            pair_scores_outfile=None,
                            print_timing=True, remove_boundary_items=False, remove_boundary_affils=False)
    compare_auc_files(new_evals_file, aucs_file_to_match)


def case6_0item_keep_0and1affils(adj_mat_infile, results_dir, aucs_file_to_match):
    print("\nCase 6\n")
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    # score matrix the way it comes: with all-0 and all-1 affils, and an item that's all-0 once the all-1 affil is gone
    # Note: that all-0 item (an induced boundary node) can't be handled quite correctly by the exp model. But it works
    # out well enough, because it ends up with a parameter very close to zero.
    new_evals_file = results_dir + "/evals-case6.txt"
    score_data.run_and_eval(adj_mat,
                            true_labels_func=expts_labeled_data.true_labels_for_expts_with_5pairs,
                            method_spec=['weighted_corr', 'weighted_corr_exp'],
                            evals_outfile=new_evals_file,
                            pair_scores_outfile=None,
                            print_timing=True, remove_boundary_items=False, remove_boundary_affils=False)
    compare_auc_files(new_evals_file, aucs_file_to_match)


# Cases 7-9: add a new item that will become all-1s after the boundary affils are removed.
# (Doing this so it'll reduce to case 1 when all boundary nodes are removed.)
def case7_01items_no_bdry_nodes(adj_mat_infile, results_dir, aucs_file_to_match):
    print("\nCase 7\n")
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    affil_degrees = np.asarray(adj_mat.sum(axis=0)).squeeze()
    adj_mat.resize((76, 206))   # orig shape was 75x206
    adj_mat[75, affil_degrees > 0] = 1  # new almost-all-1 item (preserves orig all-0 affils)

    new_evals_file = results_dir + "/evals-case7.txt"
    score_data.run_and_eval(adj_mat,
                            true_labels_func=expts_labeled_data.true_labels_for_expts_with_5pairs,
                            method_spec=['weighted_corr', 'weighted_corr_exp'],
                            evals_outfile=new_evals_file,
                            pair_scores_outfile=None,
                            print_timing=True, remove_boundary_items=True, remove_boundary_affils=True)
    compare_auc_files(new_evals_file, aucs_file_to_match)


# Cases 8-9: same data as 7, and now keep all items
def case8_01items_no_bdry_affils(adj_mat_infile, results_dir, aucs_file_to_match):
    print("\nCase 8\n")
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    affil_degrees = np.asarray(adj_mat.sum(axis=0)).squeeze()
    adj_mat.resize((76, 206))   # orig shape was 75x206
    adj_mat[75, affil_degrees > 0] = 1  # new almost-all-1 item (preserves orig all-0 affils)

    new_evals_file = results_dir + "/evals-case8.txt"
    score_data.run_and_eval(adj_mat,
                            true_labels_func=expts_labeled_data.true_labels_for_expts_with_5pairs,
                            method_spec=['weighted_corr', 'weighted_corr_exp'],
                            evals_outfile=new_evals_file,
                            pair_scores_outfile=None,
                            print_timing=True, remove_boundary_items=False, remove_boundary_affils=True)
    compare_auc_files(new_evals_file, aucs_file_to_match)


def case9_01items_keep_all(adj_mat_infile, results_dir, aucs_file_to_match):
    print("\nCase 9\n")
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    affil_degrees = np.asarray(adj_mat.sum(axis=0)).squeeze()
    adj_mat.resize((76, 206))   # orig shape was 75x206
    adj_mat[75, affil_degrees > 0] = 1  # new almost-all-1 item (preserves orig all-0 affils)
    # Note: similar to case 6, the all-1 item (induced boundary node) can't be handled by the exp model. There is no
    # max likelihood solution for this graph. In practice, the algorithm times out -- but even if it ran longer,
    # there's no good solution to converge to. The parameter for that item needs to be near-infinity, but not infinity.
    new_evals_file = results_dir + "/evals-case9.txt"
    score_data.run_and_eval(adj_mat,
                            true_labels_func=expts_labeled_data.true_labels_for_expts_with_5pairs,
                            method_spec=['weighted_corr', 'weighted_corr_exp'],
                            evals_outfile=new_evals_file,
                            pair_scores_outfile=None,
                            print_timing=True, remove_boundary_items=False, remove_boundary_affils=False)
    compare_auc_files(new_evals_file, aucs_file_to_match)


def compare_auc_files(orig_evals_file, new_evals_file, tolerance=1e-4):
    # compare evals to stored version.
    print("Checking AUCs/evals files")
    with open(orig_evals_file, 'r') as f1, open(new_evals_file, 'r') as f2:
        for line1, line2 in zip(f1, f2):
            measure1, value1 = line1.split()
            measure2, value2 = line2.split()
            if measure1[:7] == 'akaike_' or measure1 == 'numAffils':
                continue
            assert(measure1 == measure2)
            if measure1[:14] == 'loglikelihood_':  # log likelihoods are on a bigger scale, so allow more wiggle
                assert (abs(float(value1) - float(value2)) < tolerance * 1000)
            else:
                assert(abs(float(value1) - float(value2)) < tolerance)


if __name__ == "__main__":
    adj_mat_infile = "reality_appday_94/data94_adjMat.mtx.gz"
    results_dir = "reality_appday_94/python-out"
    orig_aucs_file = "reality_appday_94/results94.txt"
    case1_no_bdry_nodes(adj_mat_infile, results_dir, orig_aucs_file)
    case2_keep_0affils(adj_mat_infile, results_dir, orig_aucs_file)
    case3_keep_0and1affils(adj_mat_infile, results_dir, orig_aucs_file)

    aucs_file_0item = "reality_appday_94/results94-75items.txt"
    case4_0item_no_bdry_affils(adj_mat_infile, results_dir, aucs_file_0item)
    case5_0item_keep_0affils(adj_mat_infile, results_dir, aucs_file_0item)
    # the "more wiggle" added to loglikelihood tolerance is because params for case 6 end up slightly different
    case6_0item_keep_0and1affils(adj_mat_infile, results_dir, aucs_file_0item)

    case7_01items_no_bdry_nodes(adj_mat_infile, results_dir, orig_aucs_file)
    aucs_file_01items = "reality_appday_94/results94-76items.txt"
    case8_01items_no_bdry_affils(adj_mat_infile, results_dir, aucs_file_01items)
    # learned late: case 9 is just never going to work.
    # case9_01items_keep_all(adj_mat_infile, results_dir, aucs_file_01items)
