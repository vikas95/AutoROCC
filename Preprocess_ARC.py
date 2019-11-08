import csv
count=0
lens=[]
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')

stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
# stop_words = ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
# "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"] ## Lucene stopwords...

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

stop_words=[lmtzr.lemmatize(w1) for w1 in stop_words]
stop_words=list(set(stop_words))
"""
with open('ARC_corpus/ARC-Challenge/ARC-Challenge-Dev.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        count+=1
        lens.append(len(row))


print (set(lens))
"""
def Query_boosting_sent(query_sentences, ans_sent, boosting_factor, stop_word_flag,):
    qwords = tokenizer.tokenize(query_sentences.lower())
    qwords = [lmtzr.lemmatize(w1) for w1 in qwords]

    cand_words = tokenizer.tokenize(ans_sent.lower())
    cand_words = [lmtzr.lemmatize(w1) for w1 in cand_words]


    if stop_word_flag == 1:
        qwords = [w for w in qwords if not w in stop_words]
        cand_words = [w for w in cand_words if not w in stop_words]

    new_sent = " ".join(qwords)
    for cw in cand_words:
        new_sent = new_sent + " " + cw +"^" +str(boosting_factor)
    new_sent += "\n"
    return new_sent

def Preprocess_QA_sentences(sentences, stop_word_flag):
    words=tokenizer.tokenize(sentences.lower())
    words=[lmtzr.lemmatize(w1) for w1 in words]
    if stop_word_flag==1:
       words = [w for w in words if not w in stop_words]
    # new_sent=" ".join(words)
    # new_sent+="\n"
    return words

def Preprocess_KB_sentences(sentences, stop_word_flag):
    sentence_words = sentences.strip().split()
    BM25_score = float(sentence_words[0])
    sentences = " ".join(sentence_words[1:])
    words=tokenizer.tokenize(sentences)
    words=[lmtzr.lemmatize(w1) for w1 in words]
    if stop_word_flag==1:
       words = [w for w in words if not w in stop_words]
    # new_sent=" ".join(words)
    # new_sent+="\n"
    return BM25_score, words

def Write_ARC_KB(filename, new_file, stop_word_flag):
    KB=open(filename,"r")
    new_KB=open(new_file,"w")
    count=0
    for line in KB:
        new_sent=Preprocess_KB_sentences(line.strip(), stop_word_flag)
        new_KB.write(new_sent)
        count+=1
        if count%10000==0:
           print(count)

def get_IDF_weights(file_name, IDF):
    Doc_len=[]
    Corpus=[]
    All_words=[]
    file1=open(file_name)
    for line in file1:    ####### each line is a doc
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



class Preprocess_Arc:
    def __init__(self, ARC, directory):
        self.corpus_name=ARC
        self.directory=directory
        self.col_size=[]
        self.question=[]
        self.candidates=[]
        self.algebra_question=[]
        self.vocabulory=[]
        self.correct_ans=[]
        self.negative_ques=[]
        self.ques_ID = []
        self.ques_grade = []
        self.correct_ans_alphabet_num = []

    def preprocess(self):
        if self.corpus_name == "ARC":
            """
            file1=open(self.directory,"r")
            for line in file1:
                cols=line.split(", ")
                self.col_size.append(len(cols))
            """
            with open(self.directory, newline='') as f:
                reader = csv.reader(f)
                count=0
                for row in reader:
                    count += 1
                    if count==1:
                       pass
                    else:

                        # print (count)
                        self.ques_ID.append(row[0])
                        self.ques_grade.append(row[7])
                        self.correct_ans_alphabet_num.append(row[3])

                        self.col_size.append(len(row))
                        if row[3]=="A" or row[3]=="1":
                           self.correct_ans.append(0)
                        elif row[3]=="B" or row[3]=="2":
                           self.correct_ans.append(1)
                        elif row[3] == "C" or row[3] == "3":
                            self.correct_ans.append(2)
                        elif row[3] == "D" or row[3] == "4":
                            self.correct_ans.append(3)
                        elif row[3] == "E" or row[3] == "5":
                            self.correct_ans.append(4)
                        else:
                            print ("this is because of this naming shit", row[3])


                        question_ans=row[9]
                        question_ans_voc=tokenizer.tokenize(question_ans.lower())
                        question_ans_voc=[lmtzr.lemmatize(w1) for w1 in question_ans_voc]
                        self.vocabulory+=question_ans_voc
                        if "least" in question_ans_voc:
                            self.negative_ques.append(count-1)


                        offset=3
                        int_count=0 ## number of numerals in the question
                        for w1 in question_ans.split():
                            # self.vocabulory.add(lmtzr.lemmatize(tokenizer.tokenize(w1.lower())[0]))
                            if w1.isdigit():
                               int_count+=1
                        if int_count>1:
                           self.algebra_question.append(question_ans)

                        if "(A)" in question_ans:
                            A_index = question_ans.index("(A)")
                            B_index = question_ans.index("(B)")
                            C_index = question_ans.index("(C)")
                            if "(E)" in question_ans:
                                D_index = question_ans.index("(D)")
                                E_index = question_ans.index("(E)")

                                self.candidates.append([question_ans[A_index + offset:B_index - 1], question_ans[B_index + offset:C_index - 1],
                                     question_ans[C_index + offset:D_index - 1], question_ans[D_index + offset:E_index-1], question_ans[E_index + offset:]])
                            elif "(D)" in question_ans:
                                D_index = question_ans.index("(D)")
                                self.candidates.append([question_ans[A_index + offset:B_index - 1], question_ans[B_index + offset:C_index - 1],
                                     question_ans[C_index + offset:D_index - 1], question_ans[D_index + offset:]])

                            else:
                                self.candidates.append([question_ans[A_index + offset:B_index - 1], question_ans[B_index + offset:C_index - 1],
                                     question_ans[C_index + offset:]])



                            self.question.append((question_ans[:A_index - 1]))
                        else:
                            A_index = question_ans.index("(1)")
                            B_index = question_ans.index("(2)")
                            C_index = question_ans.index("(3)")

                            if "(5)" in question_ans:
                               E_index = question_ans.index("(5)")
                               D_index = question_ans.index("(4)")
                               self.candidates.append([question_ans[A_index + offset:B_index - 1],
                                                       question_ans[B_index + offset:C_index - 1],
                                                       question_ans[C_index + offset:D_index - 1],
                                                       question_ans[D_index + offset:E_index - 1],
                                                       question_ans[E_index + offset:]])

                            elif "(4)" in question_ans:
                                D_index = question_ans.index("(4)")
                                self.candidates.append(
                                    [question_ans[A_index + offset:B_index - 1], question_ans[B_index + offset:C_index - 1],
                                     question_ans[C_index + offset:D_index - 1], question_ans[D_index + offset:]])
                            else:
                                self.candidates.append(
                                    [question_ans[A_index + offset:B_index - 1], question_ans[B_index + offset:C_index - 1],
                                     question_ans[C_index + offset:]])

                            self.question.append((question_ans[:A_index - 1]))

            return self.ques_grade, self.question,self.candidates, self.algebra_question, set(self.vocabulory), self.correct_ans, self.ques_ID, self.correct_ans_alphabet_num

