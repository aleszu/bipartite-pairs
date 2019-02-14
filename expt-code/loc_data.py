from scipy import sparse
import random
import score_data


# Returns: sparse adjacency matrix, array storing affil name for each column
# Does not leave a blank row for people w/o data
def read_loc_adj_mat(infile, max_rows= None):
    affil_map = {}  # label --> position
    affil_array = []    # same data, but position --> label

    # we'll create the matrix at the very end. Meanwhile, store (row,col) coords.
    coord_row = []
    coord_col = []
    row_labels = []

    with open(infile, 'r') as fpin:
        fpin.readline()     # header
        for line in fpin:
            personID, affil_string = line.split(",")
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

    matrix = sparse.csc_matrix(([1] * len(coord_row), (coord_row, coord_col)))
    return matrix, row_labels


# To use for generating data sets. Read the full adj_mat, choose rows to keep,
# save just the row IDs. Also return a function that creates edge labels (for these specific rows).
# Returns: adj_mat, row labels for that matrix, function that generates true labels
def read_sample_save(adj_mat_infile, edges_infile, num_nodes, rows_outfile):
    adj_mat, row_labels = read_loc_adj_mat(adj_mat_infile)

    row_ids_to_keep = set(random.sample(range(adj_mat.shape[0]), num_nodes))
    adj_mat_to_keep = adj_mat[sorted(row_ids_to_keep),]
    row_labels_to_keep = [row_labels[i] for i in sorted(row_ids_to_keep)]  # oddly, subset notation above doesn't work

    # challenge: adj_mat_to_keep doesn't remember the old/semantically meaningful row labels. Need to keep these around
    # to send to the pair generators.

    print "Sampled " + str(num_nodes) + " nodes"
    with open(rows_outfile, 'wt') as fp:
        fp.write(" ".join(map(str, sorted(row_ids_to_keep))))   # probably need better syntax

    # edges can be stored efficiently in another sparse matrix
    label_generator = get_label_generator_from_edgefile(edges_infile, row_labels_to_keep, max(row_labels)+1)

    return adj_mat_to_keep, row_labels_to_keep, label_generator


# To read existing data sets and use for expts
def get_loc_expt_data(adj_mat_infile, edges_infile, row_ids_infile):
    adj_mat, row_labels = read_loc_adj_mat(adj_mat_infile)

    with open(row_ids_infile, 'r') as fin:
        row_ids_to_keep = sorted(map(int, fin.readline().split()))  # row ids are all on one line, space-separated

    adj_mat_to_keep = adj_mat[row_ids_to_keep,]
    row_labels_to_keep = [row_labels[i] for i in row_ids_to_keep]
    label_generator = get_label_generator_from_edgefile(edges_infile, row_labels_to_keep, max(row_labels)+1)

    return adj_mat_to_keep, row_labels_to_keep, label_generator



# tot_num_orig_rows: make this big enough to include the row labels for every edge we might keep
# (If there were people w/o affils, they won't be in orig adj_matrix, but will be rows here. On the other hand,
# edges_infile may contain yet-bigger row labels (that we didn't encounter in adj_matrix), but we can safely ignore them
# b/c we won't possibly have sampled those nodes.)
def get_label_generator_from_edgefile(edges_infile, ids_to_keep, tot_num_orig_rows):

    edge_matrix = sparse.csc_matrix((tot_num_orig_rows, tot_num_orig_rows))
    num_edges = 0
    with open(edges_infile, 'r') as fpin:
        for line in fpin:
            item1, item2 = map(int, line.split())

            if item1 in ids_to_keep and item2 in ids_to_keep:
                # make matrix symmetric by storing each entry in both directions
                if (edge_matrix[item1, item2] == 0):
                    edge_matrix[item1, item2] = 1
                    edge_matrix[item2, item1] = 1
                    num_edges += 1

    print "Found " + str(num_edges) + " edges for the rows"
    if num_edges == 0:
        return None

    def get_true_labels_given_my_edges(pairs_generator):
        return get_true_labels_loc_data(pairs_generator, edge_matrix)
    return get_true_labels_given_my_edges


def get_true_labels_loc_data(pairs_generator, edge_matrix):
    labels = []
    for (row_idx1, row_idx2, item1_name, item2_name, pair_x, pair_y) in pairs_generator:
        label = True if (edge_matrix[item1_name, item2_name] == 1) else False
        labels.append(label)
    return labels


def run_expts_loc_data(loc_data_name='brightkite', existing_data=False, inference_subdir='inference'):
    #adj_mat_infile = '/Users/lfriedl/Documents/dissertation/real-data/' + loc_data_name + '/bipartite_adj.txt'
    #adj_mat_infile = '/Users/lfriedl/Documents/dissertation/real-data/' + loc_data_name + '/bipartite_adj_round3.txt'
    #adj_mat_infile = '/Users/lfriedl/Documents/dissertation/real-data/' + loc_data_name + '/bipartite_adj_round2.txt'
    #adj_mat_infile = '/Users/lfriedl/Documents/dissertation/real-data/' + loc_data_name + '/bipartite_adj_round2_filter.txt'
    #adj_mat_infile = '/Users/lfriedl/Documents/dissertation/real-data/' + loc_data_name + '/bipartite_adj_round1.txt'
    #adj_mat_infile = '/Users/lfriedl/Documents/dissertation/real-data/' + loc_data_name + '/bipartite_adj_round1_filter.txt'
    adj_mat_infile = '/Users/lfriedl/Documents/dissertation/real-data/' + loc_data_name + '/bipartite_adj_round0_filter.txt'
    edges_infile = '/Users/lfriedl/Documents/dissertation/real-data/' + loc_data_name + '/loc-' + loc_data_name + '_edges.txt'

    exptdir = '/Users/lfriedl/Documents/dissertation/binary-ndim/' + loc_data_name + '-expts'
    for i in range(51, 61):
        rowIDs_file = exptdir + '/data' + str(i) + '.rowIDs'
        evals_outfile = exptdir + '/' + inference_subdir + '/results' + str(i) + '.txt'
        scored_pairs_outfile= exptdir + '/' + inference_subdir + '/scoredPairs' + str(i) + ".csv.gz"
        if existing_data:
            adj_mat, row_labels, label_generator = get_loc_expt_data(adj_mat_infile, edges_infile, rowIDs_file)
        else:
            adj_mat, row_labels, label_generator = read_sample_save(adj_mat_infile, edges_infile, num_nodes=500, rows_outfile=rowIDs_file)

        if label_generator is None:
            print "Found no edges; stopping"

        else:
            score_data.run_and_eval(adj_mat, true_labels_func = label_generator, method_spec="all",
                                    evals_outfile = evals_outfile,
                                    pair_scores_outfile=scored_pairs_outfile, row_labels=row_labels,
                                    print_timing=True) #, expt1=True)

if __name__ == "__main__":

    #run_expts_loc_data()
    #run_expts_loc_data(loc_data_name = 'gowalla')
    #run_expts_loc_data(existing_data=True, inference_subdir='inference_round3')
    run_expts_loc_data(existing_data=True, inference_subdir='inference_round0_filter')
