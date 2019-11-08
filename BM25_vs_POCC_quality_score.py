import json
import os
from get_subgraph import get_POC_subgraph, get_BM25_subgraph, get_POC_subgraph_BM25_filtered, get_POC_sized_BM25_subgraph
from Compute_F1 import F1_Score_just_quality, F1_Score_just_quality_question_type
import math

num_sent = 15  ## for BM25
POCC_subgraph_size = 3  ## for POCC
output_file_dir = "MultiRC_BM25_vs_POCC_justification_quality_score/"

if not os.path.exists(output_file_dir):
    os.makedirs(output_file_dir)

split_set = "dev"

# perfectly_scored_ginds = [15, 16, 66, 101, 103, 105, 137, 142, 177, 209, 210, 211, 219, 220, 227, 233, 234, 235, 256, 257, 264, 278, 287, 321, 322, 342, 373, 377, 430, 443, 444, 445, 450, 454, 460, 461, 462, 475, 476, 534, 535, 593, 632, 657, 658, 689, 693, 694, 695, 704, 712, 737, 751, 893, 897, 923, 924, 929, 936, 958, 968, 996, 997, 1000, 1013, 1030, 1036, 1064, 1065, 1095, 1096, 1112, 1120, 1121, 1124, 1138, 1140, 1153, 1158, 1160, 1297, 1347, 1348, 1350, 1362, 1363, 1371, 1372, 1385, 1387, 1449, 1465, 1468, 1469, 1470, 1508, 1514, 1515, 1516, 1564, 1566, 1567, 1635, 1638, 1639, 1641, 1642, 1656, 1686, 1722, 1741, 1745, 1751, 1779, 1787, 1791, 1802, 1803, 1862, 1909, 1910, 1911, 1917, 1937, 1938, 1939, 1940, 1946, 1947, 1948, 1951, 1954, 1955, 1968, 1971, 1975, 1988, 2003, 2054, 2060, 2064, 2065, 2066] ## look for a short passage example amongst these passages.
perfectly_scored_ginds = [264]

if split_set == "dev":
   input_file_name =  "dev_83-fixedIds.json"
   out_file_name = "dev.tsv"
   test_out_file_name = "test.tsv"
   Test_Write_file = open(output_file_dir + test_out_file_name, "w")
elif split_set == "train":
   input_file_name = "train_456-fixedIds.json"
   out_file_name = "train.tsv"

with open("MultiRC_IDF_vals.json") as json_file:
    MultiRC_idf_vals = json.load(json_file)

total_ques = 0
All_KB_passages = []
Ans_not_found_questions = []
# ques_type_indexes = {"TFYN":[], "non_verbatim":[], "verbatim":[], "found":[], "not_found":[]}  ## this is used to measure the justification quality performance for specific type of questions.
ques_type_indexes = {"TFYN":[], "non_verbatim":[], "verbatim":[]}  ## this is used to measure the justification quality performance for specific type of questions.
Write_file = open(output_file_dir + out_file_name, "w")
with open(input_file_name) as json_file:
    json_data = json.load(json_file)
    Gold_sentences_IDs = []
    Predicted_sent_IDs = []
    Predicted_sent_POCC_IDs = []

    for para_ques in json_data["data"]:
        print ("we are at this question: ", total_ques)
        current_KB_passage_sents = []
        total_ques += len(para_ques['paragraph']["questions"])
        num_of_justifications = para_ques['paragraph']["text"].count("<br>")
        # print (num_of_justifications, para_ques['paragraph']["text"])
        for i in range(num_of_justifications):
            start_index = para_ques['paragraph']["text"].find("<b>Sent "+str(i+1)+ ": </b>") + len("<b>Sent "+str(i+1)+ ": </b>")
            end_index = para_ques['paragraph']["text"].find("<b>Sent "+str(i+2)+ ": </b>")
            if i == num_of_justifications-1:
                current_KB_passage_sents.append(para_ques['paragraph']["text"][start_index:end_index].replace("<br", ""))
            else:
                current_KB_passage_sents.append(para_ques['paragraph']["text"][start_index:end_index].replace("<br>", ""))

        All_KB_passages.append(current_KB_passage_sents)

        # print (len(para_ques['paragraph']["questions"]),para_ques['id'])
        # print (para_ques['paragraph']["questions"][1],para_ques['id'])

        for qind, ques_ans1 in enumerate(para_ques['paragraph']["questions"]):
            question_text = ques_ans1['question']

            for cand_ind, cand_ans in enumerate(ques_ans1['answers']):
                GOLD_passage, pred_sent_indexes = get_BM25_subgraph(question_text, cand_ans["text"], current_KB_passage_sents, 1)


                if cand_ans['isAnswer'] == True:
                   new_line = str(1) + "\t" + para_ques['id'] + "\t" + str(qind) + "_" + str(cand_ind) + "\t" + question_text + " " + cand_ans["text"] + "\t" + GOLD_passage + "\n"
                   Gold_sentences_IDs.append(ques_ans1["sentences_used"])
                   Predicted_sent_IDs.append(pred_sent_indexes.tolist())

                   if (" ".join(current_KB_passage_sents)).lower().find(cand_ans["text"].lower()) == -1:
                   # if (" ".join(current_KB_passage_sents)).find(cand_ans["text"]) == -1:
                       if cand_ans["text"].lower() in ["yes","no","true","false"]:
                          ques_type_indexes["TFYN"].append(len(Gold_sentences_IDs))
                       else:
                          Ans_not_found_questions.append(para_ques['id'])

                          ques_type_indexes["non_verbatim"].append(len(Gold_sentences_IDs))
                   else:
                       if cand_ans["text"].lower() in ["yes", "no", "true", "false"]:
                           ques_type_indexes["TFYN"].append(len(Gold_sentences_IDs))
                       else:
                           ques_type_indexes["verbatim"].append(len(Gold_sentences_IDs))


                   # GOLD_passage_POCC_not_used, pred_sent_indexes_POCC = get_POC_subgraph_BM25_filtered(question_text, cand_ans["text"], current_KB_passage_sents, len(current_KB_passage_sents), 15)  ## care about only the correct answers
                   # GOLD_passage_POCC_not_used, pred_sent_indexes_POCC = get_POC_subgraph(question_text, cand_ans["text"], current_KB_passage_sents, max(2,math.ceil(0.2*len(current_KB_passage_sents))))  ## care about only the correct answers
                   # GOLD_passage_POCC_not_used, pred_sent_indexes_POCC, ROCC_vals = get_POC_sized_BM25_subgraph(question_text, cand_ans["text"], current_KB_passage_sents, MultiRC_idf_vals, POCC_subgraph_size, 1)  ## care about only the correct answers
                   GOLD_passage_POCC_not_used, pred_sent_indexes_POCC, ROCC_vals = get_POC_subgraph(question_text, cand_ans["text"], current_KB_passage_sents, MultiRC_idf_vals, POCC_subgraph_size, 1)  ## care about only the correct answers
                   Predicted_sent_POCC_IDs.append(pred_sent_indexes_POCC)


                   if len(Gold_sentences_IDs) - 1 == 264:
                      print ("The indexes, BM25, overlap score, ques_coverage_score, ans_coverage_score are as following: ", pred_sent_indexes_POCC, ROCC_vals)

                else:
                   new_line = str(0) + "\t" + para_ques['id'] + "\t" + str(qind) + "_" + str(cand_ind) + "\t" + question_text + " " + cand_ans["text"] + "\t" + GOLD_passage + "\n"

                Write_file.write(new_line)
                if split_set == "dev":
                   Test_Write_file.write(new_line)


print ("precision, recall and Fscores are: ", F1_Score_just_quality(Gold_sentences_IDs, Predicted_sent_IDs) )
print ("precision, recall and Fscores are: ", F1_Score_just_quality_question_type(Gold_sentences_IDs, Predicted_sent_POCC_IDs, ques_type_indexes) )


# print("Questions where answers have to be inferred rather than extracted are: ", len(Gold_sentences_IDs),len(Predicted_sent_POCC_IDs),len(Ans_not_found_questions), Ans_not_found_questions)

