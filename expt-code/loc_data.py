from __future__ import print_function
from builtins import str, map, range
from scipy import sparse
import numpy as np
import gzip

# todo: if ok, delete everything after read_loc_adj_mat()

# Returns: sparse adjacency matrix, array storing item name for each row (starting with "0"),
# array storing affil name for each column.
# Does not see, nor create blank rows for, people w/o data
def read_loc_adj_mat(infile, max_rows= None):
    affil_map = {}  # label --> position
    affil_array = []    # same data, but position --> label

    # we'll create the matrix at the very end. Meanwhile, store (row,col) coords.
    coord_row = []
    coord_col = []
    row_labels = []

    with open_poss_compressed(infile) as fpin:
        fpin.readline()     # header
        for line in fpin:
            personID, affil_string = line.rstrip().split(",")
            personID = int(personID)
            if (max_rows is not None) and (personID >= max_rows):
                break
            row_num = len(row_labels)
            row_labels.append(personID)

            for affil in affil_string.split("|"):
                affil_num = affil_map.get(affil, None)
                if affil_num is None:
                    # give it a new one!
                    affil_num = len(affil_map)
                    affil_map[affil] = affil_num
                    affil_array.append(affil)


                # put it in the matrix
                coord_row.append(row_num)
                coord_col.append(affil_num)

    matrix = sparse.csc_matrix(([1] * len(coord_row), (coord_row, coord_col)), dtype=np.int8)
    return matrix, row_labels, affil_array



# To read existing data sets and use for expts
def get_loc_expt_data(adj_mat_infile, edges_infile, row_ids_infile):
    adj_mat, row_labels, _ = read_loc_adj_mat(adj_mat_infile)

    with open(row_ids_infile, 'r') as fin:
        row_ids_to_keep = sorted(map(int, fin.readline().split()))  # row ids are all on one line, space-separated

    adj_mat_to_keep = adj_mat[row_ids_to_keep,]
    row_labels_to_keep = [row_labels[i] for i in row_ids_to_keep]
    label_generator = get_label_generator_from_edgefile(edges_infile, row_labels_to_keep)

    return adj_mat_to_keep, row_labels_to_keep, label_generator


def load_edge_matrix(edges_infile, ids_to_keep=None):
    coord_row = []
    coord_col = []
    row_labels = []         # position --> label
    row_labels_map = {}     # label --> position
    if ids_to_keep is not None:
        row_labels = ids_to_keep
        row_labels_map = {ids_to_keep[pos]:pos for pos in range(len(ids_to_keep))}

    with open(edges_infile, 'r') as fpin:
        for line in fpin:
            item1, item2 = list(map(int, line.split()))

            if ids_to_keep is None or (item1 in ids_to_keep and item2 in ids_to_keep):
                item1_num = row_labels_map.get(item1, len(row_labels_map))
                if item1_num == len(row_labels_map):
                    # we just gave it a new one!
                    row_labels_map[item1] = item1_num
                    row_labels.append(item1_num)
                item2_num = row_labels_map.get(item2, len(row_labels_map))
                if item2_num == len(row_labels_map):
                    # we just gave it a new one!
                    row_labels_map[item2] = item2_num
                    row_labels.append(item2_num)

                coord_row.append(item1_num)
                coord_col.append(item2_num)

    num_people = len(row_labels_map)
    edge_matrix = sparse.csc_matrix(([1] * len(coord_row), (coord_row, coord_col)), dtype=np.int8,
                                    shape=(num_people, num_people))
    return edge_matrix, row_labels_map


# tot_num_orig_rows: make this big enough to include the row labels for every edge we might keep
# (If there were people w/o affils, they won't be in orig adj_matrix, but will be rows here. On the other hand,
# edges_infile may contain yet-bigger row labels (that we didn't encounter in adj_matrix), but we can safely ignore them
# b/c we won't possibly have sampled those nodes.)
# Edge file starts numbering people with 0, like bipartite_adj file.
def get_label_generator_from_edgefile(edges_infile, ids_to_keep):
    edge_matrix, edge_row_labels_map = load_edge_matrix(edges_infile, ids_to_keep)
    num_edges = edge_matrix.count_nonzero()
    print("Found " + str(num_edges) + " edges for the rows")
    if num_edges == 0:
        return None

    def get_true_labels_given_my_edges(pairs_generator):
        return get_true_labels_loc_data(pairs_generator, edge_matrix, edge_row_labels_map)
    return get_true_labels_given_my_edges


def get_true_labels_loc_data(pairs_generator, edge_matrix, edge_row_labels_map):
    labels = []
    for (row_idx1, row_idx2, item1_name, item2_name, pair_x, pair_y) in pairs_generator:
        label = True if (edge_matrix[edge_row_labels_map[item1_name], edge_row_labels_map[item2_name]] == 1) else False
        labels.append(label)
    return labels


# Utility fn for possibly compressed file
open_poss_compressed = lambda f: gzip.open(f,"r") if f.endswith(".gz") else open(f)


if __name__ == "__main__":
    pass
    #run_expts_loc_data()
    #run_expts_loc_data(loc_data_name = 'gowalla')
    #run_expts_loc_data(existing_data=True, inference_subdir='inference_round3')
    # run_expts_loc_data(existing_data=True, inference_subdir='inference_round0_filter')

    # stratify_by_num_edges(adj_mat_infile='/Users/lfriedl/Documents/dissertation/real-data/brightkite/bipartite_adj.txt',
    #                       edges_infile='/Users/lfriedl/Documents/dissertation/real-data/brightkite/loc-brightkite_edges.txt',
    #                       outdir='/Users/lfriedl/Documents/dissertation/real-data/brightkite/', min_edges=10, max_edges=10)
    # run_expts_loc_data(inference_subdir='inference_10friends')