import faiss
from timeit import default_timer as timer
import sys
import score_data
import scoring_with_faiss
import loc_data



def test_faiss_basic_calls():
    adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz"
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    pi_vector_learned = score_data.learn_pi_vector(adj_mat)
    pi_vector, adj_mat = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

    # can do dot product on plain adj matrix -- just computes sharedSize
    index = faiss.IndexFlatIP(adj_mat.shape[1])    # takes numCols as arg
    # mimicking tutorial example:
    #index.add(np.random.random((100, adj_mat.shape[1])).astype('float32'))
    adj_for_faiss = adj_mat.toarray().astype('float32') # adj_mat is sparse, but faiss wants dense. and, apparently, wants float32.
    index.add(adj_for_faiss)
    print "index.is_trained: " + str(index.is_trained) + ", index.total: " + str(index.ntotal)

    # look at 10 nearest neighbors of each input
    distances10, neighbors10 = index.search(adj_for_faiss, 10)

    distances, neighbors = index.search(adj_for_faiss, adj_for_faiss.shape[0])  # all pairs
    print 'basic calls ran'


def test_score_wc_faiss():
    adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz"
    adj_mat = score_data.load_adj_mat(adj_mat_infile)
    pi_vector_learned = score_data.learn_pi_vector(adj_mat)
    pi_vector, adj_mat = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

    scores_data_frame = scoring_with_faiss.score_pairs_faiss(adj_mat, which_methods=['weighted_corr_faiss'], how_many_neighbors=-1, print_timing=True,
                      pi_vector=pi_vector)
    print 'scores look like (sample):' + str(scores_data_frame.head())
    # note for later: scores_data_frame.reset_index() makes it save item1 & item2 as regular columns, defaults back to index of row numbers

def test_faiss_plus_normal():
    adj_mat_infile = "reality_appweek_50/data50_adjMat.mtx.gz"
    adj_mat = score_data.load_adj_mat(adj_mat_infile)

    score_data.run_and_eval(adj_mat,
                            true_labels_func=score_data.true_labels_for_expts_with_5pairs,
                            # method_spec="all",
                            method_spec=['weighted_corr', 'weighted_corr_faiss'],
                            evals_outfile="reality_appweek_50/python-out/evals-test.txt",
                            print_timing=True)

# borrowed from tests.py
def resources_test(run_all_implementations=True):
    # Let's read in portions of a big matrix in increasing size, and for each size, score all pairs (both sparse and dense).
    # This will let us see how things scale and where memory limits will come in.
    infile = "/Users/lfriedl/Documents/dissertation/real-data/brightkite/bipartite_adj.txt"

    num_nodes = (100, 1000, 5000)  # my OS kills it at 10000 (due to memory)
    for num_to_try in num_nodes:
        adj_mat, _ = loc_data.read_loc_adj_mat(infile, max_rows=num_to_try)

        pi_vector_learned = score_data.learn_pi_vector(adj_mat)
        pi_vector_preproc, adj_mat_preproc = score_data.adjust_pi_vector(pi_vector_learned, adj_mat)

        # plain WC uses "transform" when dense, "terms" when sparse -- speed varies accordingly
        methods_to_run = ['weighted_corr', 'weighted_corr_faiss']

        adj_mat_preproc_dense = adj_mat_preproc.toarray()
        print "\ndense version takes up " + str(sys.getsizeof(adj_mat_preproc_dense)) + " bytes"

        start = timer()
        scores_faiss = scoring_with_faiss.score_pairs_faiss(adj_mat, methods_to_run, print_timing=True,
                                                            pi_vector=pi_vector_preproc)

        scores_data_frame = score_data.scoring_methods.score_pairs(score_data.gen_all_pairs, adj_mat_preproc_dense,
                                                                   which_methods=methods_to_run,
                                                                   pi_vector=pi_vector_preproc, back_compat=True,
                                                                   num_docs=adj_mat_preproc.shape[0],
                                                                   mixed_pairs_sims=[.01],
                                                                   print_timing=True)
        end = timer()
        print "for matrix with " + str(adj_mat_preproc.shape[0]) + " items, " + str(adj_mat_preproc.shape[1]) \
            + " affils, "
        print "ran all methods using dense matrix in " + str(end - start) + " seconds"


if __name__ == "__main__":
    test_faiss_basic_calls()
    test_score_wc_faiss()
    test_faiss_plus_normal()
    resources_test()