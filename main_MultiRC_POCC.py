import json
import os
from get_subgraph import get_POC_subgraph, get_POC_subgraph_BM25_filtered, get_POC_sized_BM25_subgraph


num_sent = 5
# output_file_dir = "MultiRC_BM25_passage_"+str(num_sent)+"/"
# output_file_dir = "MultiRC_POCC_passage_"+str(num_sent)+"SENT/"
# output_file_dir = "MultiRC_POCC_passage_Final_m_SENT_2_3_4_5_6_withIDF/"
output_file_dir = "MultiRC_POCC_passage_Final_m_SENT_2_3_4_5_withIDF_sized_BM25_exp/"

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

with open("MultiRC_IDF_vals.json") as json_file:
    MultiRC_idf_vals = json.load(json_file)

total_ques = 0
All_KB_passages = []
Write_file = open(output_file_dir + out_file_name, "w")
print ("this is the out file: ", output_file_dir + out_file_name)

with open(input_file_name) as json_file:
    json_data = json.load(json_file)

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

            # GOLD_passage_IDS = ques_ans1["sentences_used"]
            # GOLD_passage = ""
            # for id1 in GOLD_passage_IDS:
            #     GOLD_passage += current_KB_passage_sents[id1] + " "

            for cand_ind, cand_ans in enumerate(ques_ans1['answers']):
                # GOLD_passage, POCC_passage_ind = get_POC_subgraph(question_text, cand_ans["text"], current_KB_passage_sents, num_sent)
                # GOLD_passage, POCC_passage_ind, ROCC_vals_unused = get_POC_subgraph(question_text, cand_ans["text"], current_KB_passage_sents, MultiRC_idf_vals, num_sent)  ## care about only the correct answers
                GOLD_passage, POCC_passage_ind, ROCC_vals_unused = get_POC_sized_BM25_subgraph(question_text, cand_ans["text"], current_KB_passage_sents, MultiRC_idf_vals, num_sent)  ## care about only the correct answers

                if cand_ans['isAnswer'] == True:
                   new_line = str(1) + "\t" + para_ques['id'] + "\t" + str(qind) + "_" + str(cand_ind) + "\t" + question_text + " " + cand_ans["text"] + "\t" + GOLD_passage + "\n"
                else:
                   new_line = str(0) + "\t" + para_ques['id'] + "\t" + str(qind) + "_" + str(cand_ind) + "\t" + question_text + " " + cand_ans["text"] + "\t" + GOLD_passage + "\n"

                Write_file.write(new_line)
                # print ("yes, we do come here ")
                if split_set == "dev":

                   Test_Write_file.write(new_line)
