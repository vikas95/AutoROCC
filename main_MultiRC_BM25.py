import json
import os
from get_subgraph import get_POC_subgraph, get_BM25_subgraph
from Compute_F1 import F1_Score_just_quality

num_sent = 1
output_file_dir = "MultiRC_BM25_passage_"+str(num_sent)+"/"

if not os.path.exists(output_file_dir):
    os.makedirs(output_file_dir)

split_set = "dev"

if split_set == "dev":
   input_file_name =  "dev_83-fixedIds.json"
   out_file_name = "dev.tsv"
   test_out_file_name = "test.tsv"
   Test_Write_file = open(output_file_dir + test_out_file_name, "w")
elif split_set == "train":
   input_file_name = "train_456-fixedIds.json"
   out_file_name = "train.tsv"

total_ques = 0
All_KB_passages = []
Write_file = open(output_file_dir + out_file_name, "w")
with open(input_file_name) as json_file:
    json_data = json.load(json_file)
    Gold_sentences_IDs = []
    Predicted_sent_IDs = []
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
                GOLD_passage, pred_sent_indexes = get_BM25_subgraph(question_text, cand_ans["text"], current_KB_passage_sents, num_sent)

                if cand_ans['isAnswer'] == True:
                   new_line = str(1) + "\t" + para_ques['id'] + "\t" + str(qind) + "_" + str(cand_ind) + "\t" + question_text + " " + cand_ans["text"] + "\t" + GOLD_passage + "\n"
                   Gold_sentences_IDs.append(ques_ans1["sentences_used"])
                   Predicted_sent_IDs.append(pred_sent_indexes.tolist())
                else:
                   new_line = str(0) + "\t" + para_ques['id'] + "\t" + str(qind) + "_" + str(cand_ind) + "\t" + question_text + " " + cand_ans["text"] + "\t" + GOLD_passage + "\n"

                Write_file.write(new_line)
                if split_set == "dev":
                   Test_Write_file.write(new_line)


print ("precision, recall and Fscores are: ", F1_Score_just_quality(Gold_sentences_IDs, Predicted_sent_IDs) )