'''
Evaluation Script:

This file contains the code to generate the Evaluation Files. 
For the generated summaries the Rouge1, Rouge2 and RougeL Score is calculated using rouge_score
Additionally some statistics about Words/Sentences for the input, goldensource and summaries are added
The script generates 3 outputfiles
   - eval.csv: Rougescores and statistics for the summaries
   - input_stats.csv: Statistics for the input files
   - goldensource_stats.csv: Statistics for the goldensource files
'''
from rouge_score import rouge_scorer
from helper_functions import *
from nltk.tokenize import sent_tokenize

'''
Initialize rouge_scorer 
Documentation: https://github.com/google-research/google-research/tree/983925da900541f05b5df8c102906df583c65626/rouge
'''
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

'''
Function to calculate the statistics of a text file
:param folder: Folder of the file to be analyzed
:param file: File name including the filetype (example: test.txt)
:returns: List of counted words, average word length, counted sentences, average sentenc length
'''
def file_stats(folder,file):
    text = read_text(folder+file)
    words = text.split()
    word_lengths = [len(word) for word in words]
    average_word_length = sum(word_lengths) / len(words)
    sentences = sent_tokenize(text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    average_sentence_length = sum(sentence_lengths) / len(sentences)
    return [file,len(words),average_word_length,len(sentences),average_sentence_length]

# Load all files to process
inputs = read_all_files('input')
outputs = read_all_files('output')
goldensources = read_all_files('goldensource')

# Initialize variables to run the script
csv_scores = []
goldensource_stats_csv = []
input_stats_csv = []
stats_cols = ["File", "Words","Avg. Word Len","Sentences","Avg. Sentence Len"]
score_cols = ["File","R1_Precision","R1_Recall","R1_Fmeasure","R2_Precision","R2_Recall","R2_Fmeasure","RL_Precision","RL_Recall","RL_Fmeasure","Words","Avg. Word Len","Sentences","Avg. Sentence Len"]

# Evaluate and create Statistics for all summaries and goldensources
for goldensource in goldensources:
    goldensource_stats_csv.append(file_stats("goldensource/",goldensource))
    golden_text = read_text('goldensource/'+goldensource)
    id = goldensource.split(".")[1]
    predictions = list(filter(lambda x: contains_substring(x,id),outputs))
    for prediction in predictions:
        prediction_text = read_text('output/'+prediction)
        scores = scorer.score(golden_text, prediction_text)
        rouge1 = scores.get("rouge1")
        rouge2 = scores.get("rouge2")
        rougeL = scores.get("rougeL")
        csv_scores.append([prediction,rouge1.precision,rouge1.recall,rouge1.fmeasure,rouge2.precision,rouge2.recall,rouge2.fmeasure,rougeL.precision,rougeL.recall,rougeL.fmeasure]+file_stats("output/",prediction)[1:])

# Generate statistics for input files
for input in inputs:
    input_stats_csv.append(file_stats("input/",input))

# Export all statistics to .csv files
export_to_csv(input_stats_csv,stats_cols,"input_stats.csv")
export_to_csv(csv_scores,score_cols,"eval.csv")
export_to_csv(goldensource_stats_csv,stats_cols,"goldensource_stats.csv")







