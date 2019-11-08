import numpy as np
import scipy.stats
from collections import Counter
scores = [84.37, 83.93, 84.61, 84.73, 84.68] ## spanish best dev scores from 5 runs
# print (np.mean(scores))

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, h


def Stability_metric(alpha_perf, mean1, delta_perf):
    return ((alpha_perf/float(mean1)) - delta_perf)

def compute_accuracy(gold_labels, pred_labels):
    accuracy = 0
    correct_sent = []
    # print (gold_labels[0:10], pred_labels[0:10])
    for ind, val in enumerate(gold_labels):
        if gold_labels[ind] == pred_labels[ind]:
           accuracy+=1
           correct_sent.append(ind)
    return accuracy/float(len(gold_labels)), correct_sent



def F1_Score_just_quality(gold_labels, pred_labels):
    precision = 0
    recall = 0

    perfect_scored_paragraphs = []

    if len(gold_labels) == len(pred_labels):
       for gind, glabel in enumerate(gold_labels):
           precision += len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(max(1,len(pred_labels[gind])))
           recall += len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(len(gold_labels[gind]))

           if len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(max(1,len(pred_labels[gind]))) == 1 and len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(len(gold_labels[gind])) == 1:
              perfect_scored_paragraphs.append(gind)

       final_recall = recall/float(max(1,len(gold_labels)))
       final_precision = precision/float(max(1,len(gold_labels)))

    else:
       print ("The case should not happen, RECHECK")

    # print ("the final precision and recall are as following: ", final_precision, final_recall)

    return (final_precision, final_recall, (2*final_precision*final_recall)/float(max(1, final_precision+final_recall))), perfect_scored_paragraphs



def F1_Score_just_quality_question_type(gold_labels, pred_labels, question_type):
    precision = 0
    recall = 0

    perfect_scored_paragraphs = []

    precision_per_question_type = {}
    recall_per_question_type = {}

    for key1 in question_type.keys():
        precision_per_question_type.update({key1:[]})
        recall_per_question_type.update({key1:[]})

    if len(gold_labels) == len(pred_labels):
       for gind, glabel in enumerate(gold_labels):
           precision += len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(max(1,len(pred_labels[gind])))
           recall += len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(len(gold_labels[gind]))

           if len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(max(1,len(pred_labels[gind]))) == 1 and len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(len(gold_labels[gind])) == 1:
              perfect_scored_paragraphs.append(gind)

           for key1 in question_type.keys():
               if gind in question_type[key1]:
                  precision_per_question_type[key1].append(len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(max(1,len(pred_labels[gind]))))
                  recall_per_question_type[key1].append( len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(len(gold_labels[gind])) )
                  break


       final_recall = recall/float(max(1,len(gold_labels)))
       final_precision = precision/float(max(1,len(gold_labels)))
       for key1 in precision_per_question_type.keys():
           precision_per_question_type[key1] = sum(precision_per_question_type[key1]) / float(len(precision_per_question_type[key1]))
           recall_per_question_type[key1] = sum(recall_per_question_type[key1]) / float(len(recall_per_question_type[key1]))
    else:
       print ("The case should not happen, RECHECK")

    # print ("the final precision and recall are as following: ", final_precision, final_recall)

    return (final_precision, final_recall, (2*final_precision*final_recall)/float(max(1, final_precision+final_recall))), precision_per_question_type, recall_per_question_type





def F1_Score_just_quality_without_NEWS_Wiki(domain_specific_preds):
    precision = 0
    recall = 0
    perfect_scored_paragraphs = []
    total_labels = 0

    for key1 in domain_specific_preds.keys():
        if key1 in ["News", "Wiki_articles"]:
           print("Yes, we do come here, ", key1)
           pass
        else:
            gold_labels = domain_specific_preds[key1]["gold"]
            pred_labels = domain_specific_preds[key1]["BERT_pred"]

            if len(gold_labels) == len(pred_labels):
               for gind, glabel in enumerate(gold_labels):
                   precision += len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(max(1,len(pred_labels[gind])))
                   recall += len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(len(gold_labels[gind]))

                   if len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(max(1,len(pred_labels[gind]))) == 1 and len(set(gold_labels[gind]).intersection(set(pred_labels[gind])))/float(len(gold_labels[gind])) == 1:
                      perfect_scored_paragraphs.append(gind)
               total_labels += len(gold_labels)
            else:
               print ("The case should not happen, RECHECK")

    final_recall = recall / float(max(1, total_labels))
    final_precision = precision / float(max(1, total_labels))

    return (final_precision, final_recall, (2*final_precision*final_recall)/float(max(1, final_precision+final_recall))), perfect_scored_paragraphs





