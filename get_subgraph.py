from Preprocess_ARC import Preprocess_KB_sentences, Preprocess_QA_sentences
from Graph_nodes import get_all_combination_withCoverage_best_graph, get_all_combination_withCoverage_best_graph_Cand_boost, get_all_combination_withCoverage_best_graph_Cand_boost_withIDF, get_all_combination_withCoverage_best_graph_Cand_boost_ALL, \
    get_all_combination_withCoverage_best_graph_Cand_boost_withIDF_forLR
from BM25_function import get_BM25_scores
import numpy as np
import collections
from Overlap_analysis import calculate_overlap, calculate_all_overlap, calculate_overlap_labels, get_union, get_intersection, calculate_kappa
from itertools import combinations
import math
"""

Put the condition of decent sized justification also in these functions
"""

def get_POC_subgraph(ques_text, answer_text, justifications, IDF_vals, subgraph_size = 5, return_ROCC_vals = 0):  ## subgraph size is not used
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    answer_terms = Preprocess_QA_sentences(answer_text, 1)

    best_subgraph_ROCC_vals = {"indexes":[],"R":[], "O":[], "C_ans":[],"C_ques":[]}

    All_justification_terms = {}
    BM25_scores = {}
    gensim_BM25_scores = get_BM25_scores(justifications, ques_text + " " + answer_text)

    min_score = abs(min(gensim_BM25_scores))
    gensim_BM25_scores = [min_score + score_1 for score_1 in gensim_BM25_scores]

    for jind1, just1 in enumerate(justifications):
        jind_score = gensim_BM25_scores[jind1]
        just_terms = Preprocess_QA_sentences(just1,1)

        All_justification_terms.update({jind1: just_terms})  ## this is basically list of lists
        BM25_scores.update({jind1: jind_score})
    # print ("len of justificatio list is :", All_justification_terms)
    # best_subgraph = get_all_combination_withCoverage_best_graph_Cand_boost(All_justification_terms, BM25_scores, ques_terms, answer_terms, subgraph_size )
    # if return_ROCC_vals == 1:
    best_subgraph, overlap_scores, ques_coverage_scores, ans_coverage_scores = get_all_combination_withCoverage_best_graph_Cand_boost_withIDF(All_justification_terms, BM25_scores, ques_terms, answer_terms, subgraph_size, IDF_vals )

    # print ("the best subgraph that we get is ", best_subgraph)
    if best_subgraph == "Crashed":
       best_justifications = justifications # [justifications[s1] for s1 in [i+1 for i in range(subgraph_size)]]
       best_subgraph_indexes = [i for i in range(subgraph_size)]
    else:
       best_subgraph = sorted(best_subgraph)
       best_justifications = [justifications[int(s1)] for s1 in best_subgraph]
       best_subgraph_indexes = [int(s1) for s1 in best_subgraph]

       if return_ROCC_vals == 1:
          best_subgraph_ROCC_vals["indexes"] = best_subgraph_indexes
          best_subgraph_ROCC_vals["R"] = [BM25_scores[int(s1)] for s1 in best_subgraph]
          best_subgraph_ROCC_vals["C_ques"] = ques_coverage_scores
          best_subgraph_ROCC_vals["C_ans"] = ans_coverage_scores

    # if len(best_justifications)<len(justifications):
    #    print ("yep, we remove atleast few noisy sentences ", len(best_justifications), len(justifications))
    return " ".join(best_justifications), best_subgraph_indexes, best_subgraph_ROCC_vals  ## returning the whole passage




def get_POC_subgraph_LR(ques_text, answer_text, justifications, IDF_vals, All_x_features, All_y_features, gold_labels, subgraph_size = 5, return_ROCC_vals = 0):  ## subgraph size is not used
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    answer_terms = Preprocess_QA_sentences(answer_text, 1)

    best_subgraph_ROCC_vals = {"indexes":[],"R":[], "O":[], "C_ans":[],"C_ques":[]}

    All_justification_terms = {}
    BM25_scores = {}
    gensim_BM25_scores = get_BM25_scores(justifications, ques_text + " " + answer_text)

    min_score = abs(min(gensim_BM25_scores))
    gensim_BM25_scores = [min_score + score_1 for score_1 in gensim_BM25_scores]

    for jind1, just1 in enumerate(justifications):
        jind_score = gensim_BM25_scores[jind1]
        just_terms = Preprocess_QA_sentences(just1,1)

        All_justification_terms.update({jind1: just_terms})  ## this is basically list of lists
        BM25_scores.update({jind1: jind_score})
    # print ("len of justificatio list is :", All_justification_terms)
    # best_subgraph = get_all_combination_withCoverage_best_graph_Cand_boost(All_justification_terms, BM25_scores, ques_terms, answer_terms, subgraph_size )
    # if return_ROCC_vals == 1:
    best_subgraph, overlap_scores, ques_coverage_scores, ans_coverage_scores, All_x_features, All_y_features = get_all_combination_withCoverage_best_graph_Cand_boost_withIDF_forLR(All_justification_terms, BM25_scores, ques_terms, answer_terms, subgraph_size, IDF_vals, All_x_features, All_y_features, gold_labels )

    # print ("the best subgraph that we get is ", best_subgraph)
    if best_subgraph == "Crashed":
       best_justifications = justifications # [justifications[s1] for s1 in [i+1 for i in range(subgraph_size)]]
       best_subgraph_indexes = [i for i in range(subgraph_size)]
    else:
       best_subgraph = sorted(best_subgraph)
       best_justifications = [justifications[int(s1)] for s1 in best_subgraph]
       best_subgraph_indexes = [int(s1) for s1 in best_subgraph]

       if return_ROCC_vals == 1:
          best_subgraph_ROCC_vals["indexes"] = best_subgraph_indexes
          best_subgraph_ROCC_vals["R"] = [BM25_scores[int(s1)] for s1 in best_subgraph]
          best_subgraph_ROCC_vals["C_ques"] = ques_coverage_scores
          best_subgraph_ROCC_vals["C_ans"] = ans_coverage_scores

    # if len(best_justifications)<len(justifications):
    #    print ("yep, we remove atleast few noisy sentences ", len(best_justifications), len(justifications))
    return " ".join(best_justifications), best_subgraph_indexes, best_subgraph_ROCC_vals, All_x_features, All_y_features  ## returning the whole passage





def get_POC_sized_BM25_subgraph(ques_text, answer_text, justifications, IDF_vals, subgraph_size = 5, return_ROCC_vals = 0):  ## subgraph size is not used
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    answer_terms = Preprocess_QA_sentences(answer_text, 1)

    best_subgraph_ROCC_vals = {"indexes":[],"R":[], "O":[], "C_ans":[],"C_ques":[]}

    All_justification_terms = {}
    BM25_scores = {}
    gensim_BM25_scores = get_BM25_scores(justifications, ques_text + " " + answer_text)

    min_score = abs(min(gensim_BM25_scores))
    gensim_BM25_scores = [min_score + score_1 for score_1 in gensim_BM25_scores]

    BM25_sorted_indexes = list(np.argsort(gensim_BM25_scores))[::-1]

    for jind1, just1 in enumerate(justifications):
        jind_score = gensim_BM25_scores[jind1]
        just_terms = Preprocess_QA_sentences(just1,1)

        All_justification_terms.update({jind1: just_terms})  ## this is basically list of lists
        BM25_scores.update({jind1: jind_score})
    # print ("len of justificatio list is :", All_justification_terms)
    # best_subgraph = get_all_combination_withCoverage_best_graph_Cand_boost(All_justification_terms, BM25_scores, ques_terms, answer_terms, subgraph_size )
    # if return_ROCC_vals == 1:
    best_subgraph, overlap_scores, ques_coverage_scores, ans_coverage_scores = get_all_combination_withCoverage_best_graph_Cand_boost_withIDF(All_justification_terms, BM25_scores, ques_terms, answer_terms, subgraph_size, IDF_vals )

    # print ("the best subgraph that we get is ", best_subgraph)
    if best_subgraph == "Crashed":
       print ("do we ever come here: ")
       best_justifications = justifications # [justifications[s1] for s1 in [i+1 for i in range(subgraph_size)]]
       best_subgraph_indexes = [i for i in range(subgraph_size)]
    else:
       best_subgraph = sorted(best_subgraph)
       best_justifications = [justifications[int(s1)] for s1 in BM25_sorted_indexes[:len(best_subgraph)]]
       best_subgraph_indexes = [int(s1) for s1 in best_subgraph]

       if return_ROCC_vals == 1:
          best_subgraph_ROCC_vals["indexes"] = best_subgraph_indexes
          best_subgraph_ROCC_vals["R"] = [BM25_scores[int(s1)] for s1 in best_subgraph]
          best_subgraph_ROCC_vals["C_ques"] = ques_coverage_scores
          best_subgraph_ROCC_vals["C_ans"] = ans_coverage_scores

    # if len(best_justifications)<len(justifications):
    #    print ("yep, we remove atleast few noisy sentences ", len(best_justifications), len(justifications))
    return " ".join(best_justifications), BM25_sorted_indexes[:len(best_subgraph)], best_subgraph_ROCC_vals  ## returning the whole passage


def get_POC_subgraph_BM25_filtered(ques_text, answer_text, justifications, subgraph_size = 5, BM25_threshold = 15):  ## subgraph size is not used
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    answer_terms = Preprocess_QA_sentences(answer_text, 0)

    All_justification_terms = {}
    BM25_scores = {}
    gensim_BM25_scores = get_BM25_scores(justifications, ques_text + " " + answer_text)

    min_score = abs(min(gensim_BM25_scores))
    gensim_BM25_scores = [min_score + score_1 for score_1 in gensim_BM25_scores]

    BM25_ranked_indexes = np.argsort(gensim_BM25_scores)[:min(len(justifications),BM25_threshold)]

    for jind1, just1 in enumerate(justifications):
        if jind1 in BM25_ranked_indexes:
            jind_score = gensim_BM25_scores[jind1]
            just_terms = Preprocess_QA_sentences(just1,1)

            All_justification_terms.update({jind1: just_terms})  ## this is basically list of lists
            BM25_scores.update({jind1: jind_score})
    # print ("len of justificatio list is :", All_justification_terms)
    best_subgraph = get_all_combination_withCoverage_best_graph_Cand_boost(All_justification_terms, BM25_scores, ques_terms, answer_terms, subgraph_size )
    # print ("the best subgraph that we get is ", best_subgraph)
    if best_subgraph == "Crashed":
       best_justifications = justifications # [justifications[s1] for s1 in [i+1 for i in range(subgraph_size)]]
       best_subgraph_indexes = [i for i in range(subgraph_size)]
    else:
       best_justifications = [justifications[int(s1)] for s1 in best_subgraph]
       best_subgraph_indexes = [int(s1) for s1 in best_subgraph]
    # if len(best_justifications)<len(justifications):
    #    print ("yep, we remove atleast few noisy sentences ", len(best_justifications), len(justifications))
    return " ".join(best_justifications), best_subgraph_indexes  ## returning the whole passage



def get_POC_subgraph_all(ques_text, answer_text, justifications, min_length):
    ques_terms = Preprocess_QA_sentences(ques_text, 1)
    answer_terms = Preprocess_QA_sentences(answer_text, 0)

    All_justification_terms = {}
    BM25_scores = {}
    for jind1, just1 in enumerate(justifications):
        jind_score, just_terms = Preprocess_KB_sentences(just1,1)

        if len(just_terms) >= min_length:
           All_justification_terms.update({jind1: just_terms})  ## this is basically list of lists
           BM25_scores.update({jind1: jind_score})
    # print ("len of justificatio list is :", All_justification_terms)
    best_subgraph = get_all_combination_withCoverage_best_graph_Cand_boost_ALL(All_justification_terms, BM25_scores, ques_terms, answer_terms)
    # print ("the best subgraph that we get is ", best_subgraph)
    if best_subgraph == "Crashed":
       best_justifications = [justifications[s1] for s1 in [i+1 for i in range(4)]] ## since 4 is the ideal size
    else:
       best_justifications = [justifications[int(s1)] for s1 in best_subgraph]
    return best_justifications



def get_BM25_subgraph(ques_text, answer_text, justifications, subgraph_size):  ## subgraph size is not used

    gensim_BM25_scores_indexes = np.argsort(get_BM25_scores(justifications, ques_text + " " + answer_text))[::-1]
    top_ranked_passage = ""
    for i in gensim_BM25_scores_indexes[:subgraph_size]:
        top_ranked_passage += justifications[i] + " "

    return top_ranked_passage, gensim_BM25_scores_indexes[:subgraph_size]  ## returning the whole passage
