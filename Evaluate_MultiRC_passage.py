import json
import os
from multirc_daniel.multirc_materials.multirc_measures import Measures
import numpy as np
# input_file = open("dev_83-fixedIds.json","r")
# input_file = open("train_456-fixedIds.json","r")

input_file_dir = "MultiRC_POCC_passage_Final_m_SENT_2_3_4_5_withIDF_Cov_ans/"

measures = Measures()

def eval_function(predictions, inputFile="dev_83-fixedIds.json"):
    input = json.load(open(inputFile))
    output = predictions
    output_map = dict([[a["pid"] + "==" + a["qid"], a["scores"]] for a in output])

    assert len(output_map) == len(output), "You probably have redundancies in your keys"

    [P1, R1, F1m] = measures.per_question_metrics(input["data"], output_map)
    print("Per question measures (i.e. precision-recall per question, then average) ")
    print("\tP: " + str(P1) + " - R: " + str(R1) + " - F1m: " + str(F1m))

    EM0 = measures.exact_match_metrics(input["data"], output_map, 0)
    EM1 = measures.exact_match_metrics(input["data"], output_map, 1)
    print("\tEM0: " + str(EM0))
    print("\tEM1: " + str(EM1))

    [P2, R2, F1a] = measures.per_dataset_metric(input["data"], output_map)

    print("Dataset-wide measures (i.e. precision-recall across all the candidate-answers in the dataset) ")
    print("\tP: " + str(P2) + " - R: " + str(R2) + " - F1a: " + str(F1a))


orig_file_input = open(input_file_dir+"test.tsv","r").readlines()

num_runs = [1,2,3]

for run1 in num_runs:
    predictions = open(input_file_dir+"Prediction_files_large/results_"+str(run1)+".tsv","r").readlines()
    All_ques_pred = [{"pid":"News/CNN/cnn-3b5bbf3ba31e4775140f05a8b59db55b22ee3e63.txt","qid":"0","scores":[1]}]
    # prev_qid = "0"
    for ind1, line in enumerate(orig_file_input[1:]): ## because I am missing the first line while feeding into BERT
        tokens1 = line.strip().split("\t")
        if tokens1[1] == All_ques_pred[-1]["pid"]:
           if tokens1[2].split("_")[0] == All_ques_pred[-1]["qid"]:
              All_ques_pred[-1]["scores"].append(int(np.argmax([ float(predictions[ind1].split()[0]), float(predictions[ind1].split()[1]) ] ) ))
           else:
              All_ques_pred.append({"pid":tokens1[1], "qid":tokens1[2].split("_")[0], "scores": [ int(np.argmax( [ float(predictions[ind1].split()[0]), float(predictions[ind1].split()[1]) ] ) ) ] })
        else:
            All_ques_pred.append({"pid": tokens1[1], "qid": tokens1[2].split("_")[0], "scores": [ int(np.argmax([float(predictions[ind1].split()[0]), float(predictions[ind1].split()[1])]) )] })

    with open('multirc_daniel/baseline-scores/POCC_5SENT_results.json', 'w') as outfile:
        json.dump(All_ques_pred, outfile)
    eval_function(All_ques_pred)
