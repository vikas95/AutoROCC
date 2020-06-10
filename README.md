# AutoROCC
Implementation of the AutoROCC paper[https://www.aclweb.org/anthology/D19-1260.pdf] published at EMNLP-IJCNLP 2019


Please pardon the variable names for now, I will update the variable names which make more sense in next 1-2 weeks. 


## Running Experiments:

1] Please download the MultiRC dataset from https://github.com/CogComp/multirc
The train and dev set are available in the above link and test set is hidden. 

2] Please run "BM25_vs_POCC_quality_score.py" to get the comparison of BM25 VS AutoROCC for justification selection performance on dev set.  

3]Please run "main_MultiRC_POCC.py" for generating MRPC format input data. After converting the MultiRC data to MRPC format (where text1 = question + candidate_answer text and text2 = justification set text), one can run BERT (https://github.com/google-research/bert) or any of the latest transformer networks (XLnet or RoBERTa) from huggingface library - https://github.com/huggingface/transformers

Please note that we have reported numbers from original BERT tensorflow codes(https://github.com/google-research/bert). Some improvements can be expected from RoBERTa or XLnet. 

4] The predictions from BERT (or XLnet or RoBerta) saved in result.tsv should be given to "Evaluate_MultiRC_passage.py".
Please note that we use MultiRC official evaluation codes within this file. 



