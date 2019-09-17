import gzip
from builtins import str

import numpy as np

# Convention for all my expt data: the true pairs are items (1,2), (3,4), etc.
# These two functions each take a generator object and return a list.
def get_true_labels_expt_data(num_true_pairs, pairs_generator):
    labels = []
    for (row_idx1, row_idx2, _, _, _, _) in pairs_generator:
        label = True if (row_idx2 < 2 * num_true_pairs and row_idx1 == row_idx2 - 1 and row_idx2 % 2) else False
        labels.append(label)
    return labels


def true_labels_for_expts_with_5pairs(pairs_generator):
    return get_true_labels_expt_data(5, pairs_generator)


def load_pi_from_file(pi_vector_infile_gz):
    pi_saved = []
    with gzip.open(pi_vector_infile_gz, 'r') as fpin:
        for line in fpin:
            pi_saved.append(float(line.strip()))
    return np.array(pi_saved)  # making type consistent with learn_pi_vector


# Keeping this (older) version around only because of "expt1" and because I haven't updated tests that call it.
# Always: remove exact 0s and 1s from columns of data + pi_vector.
# Optionally: "flip" high p's -- i.e., swap 1's and 0's in the data so that resulting p's are <= .5.
# expt1: remove affils with 0 or even 1 person attached
def adjust_pi_vector(pi_vector, adj_mat, flip_high_ps=False, expt1 = False, report_boundary_items=True):
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

    if report_boundary_items:
        rowsums = np.asarray(adj_mat_mod.sum(axis=1)).squeeze()
        num_all0 = np.sum(rowsums==0)
        num_all1 = np.sum(rowsums==adj_mat_mod.shape[1])
        if num_all0 > 0 or num_all1 > 0:
            print("(Keeping the: " + str(num_all0) + " all-0 items and " + str(num_all1) + " all-1 items)")

    return pi_vector_mod, adj_mat_mod.tocsr()