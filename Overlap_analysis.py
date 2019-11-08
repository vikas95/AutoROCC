
import numpy as np
import collections
from sklearn.metrics import cohen_kappa_score

def calculate_overlap_labels(list1, list2):
    return sum([1 if list1[ind]==list2[ind] else 0 for ind in range(len(list1))]) / float(max(   len(list1), len(list2) ) )



def calculate_overlap(list1, list2):
    # print ("The list values are: ",list2, list1)
    return  len(set(list2).intersection(set(list1))) / float(max(   len(list1), len(list2) ) )

def calculate_overlap_QA_terms(list1, list2, QA_terms):
    # print ("The list values are: ",list2, list1)
    return  len(set(list2) & set(list1) & set(QA_terms)) / float(max(   len(list1), len(list2) ) )


def calculate_kappa(list1, list2):
    return cohen_kappa_score(list1, list2)

def calculate_all_overlap(list1, list2, list3):
    return (  len(   set(list3).intersection( set(list2).intersection(set(list1)) )    )    )


def get_union(list1, list2):
    return list(set(list2).union(set(list1)))

def get_intersection(list1, list2):
    return list(set(list2).intersection(set(list1)))

def get_intersection_withIDF(list1, list2, IDF_vals):
    covered_terms = list(set(list2).intersection(set(list1)))
    # covered_terms = [t1 for t1 in list2 if t1 in list1] ## this was to check whether TF in coverage makes any difference or not. IT DOES NOT.

    covered_terms_sum = 0
    for ct1 in covered_terms:
        if ct1 in IDF_vals:
           covered_terms_sum += IDF_vals[ct1]
        else:
           covered_terms_sum += 3
           print ("this IDF value not found case should not come. ")
    return covered_terms_sum

def get_union_dummy(list1, list2):
    return [1 if list1[ind]+list2[ind] >= 1 else 0 for ind in range(len(list1))]

def get_intersection_dummy(list1, list2):
    return [1 if list2[ind]==1 and list1[ind] == list2[ind] else 0 for ind in range(len(list1))]

