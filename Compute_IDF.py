import math, json
from Preprocess_ARC import Preprocess_QA_sentences


def get_IDF_weights(all_sentences, IDF):
    Doc_len=[]
    Corpus=[]
    All_words=[]

    for line in all_sentences:    ####### each line is a doc
        line=line.lower()
        words=line.split()
        # words=tokenizer.tokenize(line)
        # words = [lmtzr.lemmatize(w1) for w1 in words]
        Document={}  ########## dictionary - having terms as key and TF as values of the key.
        Doc_len.append(len(words))
        unique_words=list(set(words))
        for w1 in unique_words:
            if w1 in IDF.keys():
               IDF[str(w1)]+=1
               # print ("yes, we come here", w1)
            #else:
               #IDF.update({str(w1):1})


        All_words += unique_words
        for term1 in unique_words:
            Document[str(term1)]=words.count(term1)

        Corpus.append(Document)
    All_words=list(set(All_words))
    return Doc_len, Corpus, All_words, IDF


def Write_IDF_vals(All_words, All_sentences, file_name):
    IDF = {}
    for each_word in All_words:
        IDF[str(each_word)] = 0

    print ("vocab len should be same", len(IDF))
    Doc_lengths, All_Documents, AW1, IDF2 = get_IDF_weights(All_sentences, IDF)

    print (len(IDF2))
    for terms_TF in All_Documents:
        for tf_key in terms_TF:
            terms_TF[tf_key] = 1 + math.log(terms_TF[tf_key])

    Total_doc = len(All_Documents)
    # Avg_Doc_len = sum(Doc_lengths) / float(len(Doc_lengths))

    for each_word in All_words:
        doc_count = IDF2[str(each_word)]

        IDF[str(each_word)] = math.log10((Total_doc - doc_count + 0.5) / float(doc_count + 0.5))

    with open(file_name, 'w') as outfile:
        json.dump(IDF, outfile)


input_files =  ["train_456-fixedIds.json", "dev_83-fixedIds.json"]

All_KB_passages = []
Vocab = []

for input_file_name in input_files:
    with open(input_file_name) as json_file:
        json_data = json.load(json_file)

        for para_ques in json_data["data"]:
            current_KB_passage_sents = []
            num_of_justifications = para_ques['paragraph']["text"].count("<br>")
            # print (num_of_justifications, para_ques['paragraph']["text"])
            for i in range(num_of_justifications):
                start_index = para_ques['paragraph']["text"].find("<b>Sent "+str(i+1)+ ": </b>") + len("<b>Sent "+str(i+1)+ ": </b>")
                end_index = para_ques['paragraph']["text"].find("<b>Sent "+str(i+2)+ ": </b>")
                if i == num_of_justifications-1:
                    current_KB_passage_sents.append(para_ques['paragraph']["text"][start_index:end_index].replace("<br", ""))
                else:
                    current_KB_passage_sents.append(para_ques['paragraph']["text"][start_index:end_index].replace("<br>", ""))

            All_KB_passages+=current_KB_passage_sents
            for sent1 in current_KB_passage_sents:
                Vocab += Preprocess_QA_sentences(sent1, 0)

Vocab = list(set(Vocab))


Write_IDF_vals(Vocab, All_KB_passages, "MultiRC_IDF_vals.json")