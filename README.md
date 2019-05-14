
# Bipartite-pairs overview

This is research code: new methods for computing similarities between nodes (within the same group) of a bipartite graph. The code computes these similarity methods -- here, named `weighted_corr_exp`, `weighted_corr`, and `mixed_pairs` -- as well as a slew of comparison methods, such as `jaccard`, `cosine`, `hamming`, etc. It takes as input the adjacency matrix of a bipartite graph, and it outputs a score (per method) for each pair.

I plan to publish this work soon. A slightly outdated writeup is found in Chapter 4 of my dissertation, [Friedland 2016](https://scholarworks.umass.edu/dissertations_2/845/). Since then, I've updated to a better null model of the graph -- this is what [`weighted_corr_exp`](#weighted_corr_exp) uses -- and found a framing that explains the (not originally anticipated) connection between `weighted_corr` and `mixed_pairs`. Work in progress!

My original code base was in R, but this repo is rewritten in python, for speed. For fast distance calculations, it relies on outside libraries where possible -- namely, [scipy's `sklearn`](https://scikit-learn.org) and [Facebook's `Faiss`](https://github.com/facebookresearch/faiss).

### Table of Contents
<!-- placeholders for TOC, using gh-md-toc --!>
<!--ts-->
   * [Bipartite-pairs overview](#bipartite-pairs-overview)
   * [Setup and dependencies](#setup-and-dependencies)
   * [Usage: commands to call](#usage-commands-to-call)
   * [Similarity scores: notes and details](#similarity-scores-notes-and-details)
   * [Implementation details](#implementation-details)

<!-- Added by: lfriedl, at: Tue May 14 19:22:16 EDT 2019 -->

<!--te-->

# Setup and dependencies
The code should work under both python2 and python3. (Main development is in python2.7; I periodically run it through [`futurize`](http://python-future.org) to keep it compatible with python3.)  If you see an ImportError about the package "builtins", it can be fixed by running `pip install future`.

External dependencies (can be installed with pip): numpy, pandas, scipy, scikit-learn, future.

If the package Faiss (<https://github.com/facebookresearch/faiss>) is installed, this package will use it and be much faster. (Faiss contains methods for fast approximate all-pairs similarity calculations. Eventually, I'll use these to _avoid_ looking at all pairs. For now, the all-pairs implementation, which they treat as a "brute force" baseline, works great for my needs.) Faiss can be installed using its [conda](https://anaconda.org/pytorch/faiss-cpu) or [pip](https://pypi.org/project/faiss/) bundles.


# Usage: commands to call
Only files under `python-scoring` should be needed by most users. The main function(s) to call are in `score_data.py`. 

Ways to call it:

* At the command line, from the directory `bipartite-pairs/python-scoring`, run
    ```
    python score_data.py adj_matrix.mtx.gz pair_scores_out.csv.gz method1 [method2 ...]
    ```
    where:  
    *  `adj_matrix.mtx.gz` is a file containing the adjacency matrix of an undirected bipartite graph, in [matrix market](https://math.nist.gov/MatrixMarket/formats.html) format. (Many languages, including python and R, have routines to read and write this format.) The graph will be treated as unweighted.
    * `pair_scores_out.csv.gz` will be the output file. It's a gzipped csv file, with one line per pair of nodes. The columns named "item1" and "item2" index the pair, and the remaining columns are the (symmetric) similarity scores.
    * `method1` and any other methods are taken from the list `all_defined_methods` defined at the top of `scoring_methods.py`. Currently these include `jaccard`, `cosine`, `cosineIDF`, `shared_size`, `hamming`, `pearson`,
                       `shared_weight11`, `shared_weight1100`, `adamic_adar`, `newman`, `mixed_pairs` (n.b. requires another argument, so can't be called through these APIs),
                       `weighted_corr`, `weighted_corr_exp`.
* From a python script, ```import score_data```,
then call one of:

  * `score_data.score_only()`. Same as the command-line call, with more optional parameters. The command-line call above is equivalent to:     
    ```
    score_data.score_only('adj_matrix.mtx.gz', ['method1', 'method2'], 'pair_scores_out.csv.gz')
    ```
  * `score_data.write_item_likelihoods()`. This does *not* score pairs. Instead, it fits a bipartite graph model to the data -- in fact, two graph models: multiple Bernoulli, and exponential. Then it computes the log likelihood, under the model, for each item. It writes a csv file with 1 line per item, and 2 log likelihoods per line.
  * `score_data.get_item_likelihoods()`. Almost the same as `score_data.write_item_likelihoods()`, but returns the log likelihoods as a list.
  * `score_data.run_and_eval()`. This is for running experiments, where you have "true pairs" to compare the scores against. Does everything from `score_only()`, plus evaluations.


# Similarity scores: notes and details
1. In my terminology, the bipartite graph provided as input describes the edges between "items" and "affiliations." "Items" are the primary objects we care about. We can compute their individual likelihoods (`score_data.write_item_likelihoods()`), and we can compute scores between (all) pairs of items.

1. Not all methods are defined if the adjacency matrix has a row or column that's all 0s or 1s (i.e., if any node has 0 edges or all possible edges). To prevent problems, the code removes any such nodes from the adjacency matrix at the very start.
   * This could mean the code returns scores for fewer items than you sent in. It will in fact screw up your labeling of them (referring to them as "0" through "number of items it keeps") unless you send in a list of `row_labels`.
   * I'm not thrilled about silently manipulating the adjacency matrix like this. I've actually found a perfectly reasonable score to use for every method -- e.g., define jaccard(0,0) = cosine(0,0) = 0. But I haven't found a good way to handle NaNs in the calculations, especially for the graph models, so this will have to do for now. (Might change later.)
   
1. For descriptions and references to most methods, see [Friedland 2016](https://scholarworks.umass.edu/dissertations_2/845/), especially Table 4.3 on p.78.
   * Note on `shared_size` (a.k.a. number of common neighbors) and `hamming` distance. By default, I convert them to be similarity measures between 0 and 1. That is, `shared_size_reported := shared_size / num affiliations`, and `hamming_reported := (num affiliations - hamming) / num affiliations`. Sometimes we'd prefer the usual old integer; the parameter `integer_ham_ssize=True` makes that happen.

1. What's this new method <a name="weighted_corr_exp"></a>`weighted_corr_exp`? It's the same as `weighted_corr`, except with an exponential model fit to the graph (to produce the estimates of P(edge)), instead of a Bernoulli model. References on this exponential model:
    * It's one of the simplest types of [Exponential random graph models (ERGMS)](https://en.wikipedia.org/wiki/Exponential_random_graph_models). The only features we use are the degrees of nodes, no other graph structure.
    * The earliest versions of ERGMs were known as p1 models. The one here is an instance of the simplest *bipartite* ERGM, which [Iacobucci and Wasserman, 1990](https://doi.org/10.1007/BF02294618) introduced as the p2 model. Another early reference on bipartite ERGMs is by [Skvoretz and Faust, 2002](https://doi.org/10.1111/0081-1750.00066).
    * The implementation I'm using to fit the data (in `python-scoring/bipartite_fitting.py:learn_biment()`) comes from [Dianati 2016](https://arxiv.org/abs/1607.01735), which derived the same model using a maximum entropy approach. 
    * Recently, this model has also been renamed the Beta-model; see, e.g., [Rinaldo, Petrovic and Fienberg 2013](https://doi.org/10.1214/12-AOS1078).

# Implementation details

* **Overview of code files**
  * `bipartite_fitting.py` and `bipartite_likelihood.py`: the Bernoulli and exponential graph models.
  * `magic_dictionary.py`: a glorified dictionary (or hashmap) storing the results. Since RAM quickly becomes a bottleneck when we're computing scores for all pairs, if there are enough pairs we store the scores in memmap files in a temporary directory. (There is a speed downside for doing this; copying the computed scores out of these files to the final output file then takes non-trivial time.)
  * `score_data.py` holds the the main wrapper functions, described in [Usage](#usage) above. It also has some data preprocessing functions (loading the adjacency matrix, getting rid of columns and rows that are all 0s or 1s).  
    It also contains generator functions for looping over pairs of nodes (`gen_all_pairs()` and `ij_gen()`). (These functions are complicated methods to do something simple: loop over all pairs of nodes. The idea is that in the future, if we're only scoring some pairs of nodes, we would define a different function to loop over only them. Then we would pass that function around instead of `gen_all_pairs()`.)
  * `scoring_methods.py` does the bulk of the work in its function `score_pairs()`, which takes a list of methods. If faiss is enabled, it sends some of the methods off to be done in `scoring_with_faiss.py`. These files contain code for computing the similarity functions. Additional code for computing the similarity functions is in `scoring_methods_fast.py`, `transforms_for_dot_products.py`, and `extra_implementations.py`.
 
* **Dot product scoring methods.** Many similarity methods can be (computed faster when) rewritten as the dot product of rows X and Y of some transformed version of the adjacency matrix. 

  For instance, (raw) `shared_size` requires no transformation; the number of common neighbors of rows X and Y of the adjacency matrix can be computed directly as X.dot(Y).
  
  As another example, say we want to compute `shared_size / num affiliations` (the default version of `shared_size`). We can transform the adjacency matrix, multiplying each entry by `1/sqrt(num affiliations)`. Taking the dot product of the resulting rows X and Y removes the `sqrt` to leave the desired answer.
  
  The function `scoring_methods.py:compute_scores_with_transform()` is a template for methods of this form. You hand it the transformation function for the adjacency matrix, and it does the rest (looping over all pairs). The transformation functions are defined in `transforms_for_dot_products.py`.

* **`Compute_scores_from_terms` scoring methods.** As shown in Table 4.3 referenced above, most similarity methods can be specified in the form: (1/denom) * sum (over affilations j) `f(Xj, Yj, pj)`. Here, affiliations are columns of the adjacency matrix, X and Y are the rows whose similarity we're computing, Xj and Yj are their respective entries in the jth column, and pj is a different probability associated with each affiliation.  
  
  That function `f(Xj, Yj, pj)` is one of:
  *  `f00(pj)`, if Xj = Yj = 0
  *  `f11(pj)`, if Xj = Yj = 1
  *  `f10(pj)`, if Xj = 1 and Yj = 0 OR Xj = 0 and Yj = 1.
  
  The simplest (and usually slow) implementations of the scoring methods:
    *  ask for the values of f00, f11, and f10 (as row vectors, covering all values of j)
    *   figure out which components of X and Y are 00, 11, or 10 (respectively)
    *   and compute the similarity measure by taking a sum of  dot products. See, for example, `extra_implementations.py:weighted_corr()`.
  
  `scoring_methods.py:compute_scores_from_terms()` is a template function on that principle. You hand it the f00, f11 and f10 functions (for a given scoring method), and it performs the whole calculation (looping over all pairs).

* **The package faiss** performs similarity calculations that can be specified as dot products. Therefore, all the dot product methods can be run via faiss. The function `scoring_with_faiss.py:compute_faiss_dotprod_distances()` is the analog of `scoring_methods.py:compute_scores_with_transform()`.

  Similarly, the "from terms" calculation can be represented as the sum of two dot products, and therefore also run through faiss. The function `scoring_with_faiss.py:compute_faiss_terms_scores()` is the analog of `scoring_methods.py:compute_scores_from_terms()`. 

* **Code walk-through when computing the similarity method `weighted_corr_exp`**. A stack trace would show something like the following functions being called.
    1. `score_data.py:score_pairs()` = main call
    2. `load_adj_mat()` and `remove_boundary_nodes()` = data preprocessing  
    3. `learn_graph_models()` = fit exponential model to data
    4. `scoring_methods.score_pairs()` = computes scores, method by method
    5. `magic_dictionary.make_me_a_dict()` = allocate storage object
    6. `separate_faiss_methods()` = figure out if we're using faiss
    7. `scoring_with_faiss.score_pairs_faiss_all_exact()` --> `score_pairs_faiss()` = entering faiss's analog of `scoring_methods.score_pairs()`
    8. Compute `weighted_corr_exp` as a dot product:
    
    ```
    if 'weighted_corr_exp_faiss' in which_methods:
      ...
      compute_faiss_dotprod_distances(adj_matrix, 
           transforms_for_dot_prods.wc_exp_transform, ...)
    ```
    
    9. Inside `compute_faiss_dotprod_distances()`, transform the adjacency matrix using the function `transforms_for_dot_prods.wc_exp_transform()`: `transformed_mat[:] = transf_func(adj_matrix.toarray() ...)`
    11. Put `transformed_mat` into a faiss inner product index: `faiss_index = ...`
    3. Compute the N nearest neighbors to each node in the data set: `faiss_index.compute_distance_subset()`. We use N = num nodes because we want all pairs of distances.
    4. The remaining step is to save the data to the outfile, at the end of `scoring_methods.score_pairs()`: `scores_storage.to_csv_gz()`.

