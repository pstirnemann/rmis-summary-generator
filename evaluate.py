from rouge_score import rouge_scorer
from helper_functions import *
import pandas as pd
from nltk.tokenize import sent_tokenize

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

def contains_substring(string, substring):
    return substring in string

def export_to_csv(list,cols,file_name):
    df = pd.DataFrame(list,columns=cols)
    df.to_csv(file_name,index=False,encoding='utf-8')

def file_stats(folder,file):
    text = read_text(folder+file)
    words = text.split()
    word_lengths = [len(word) for word in words]
    average_word_length = sum(word_lengths) / len(words)
    sentences = sent_tokenize(text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    average_sentence_length = sum(sentence_lengths) / len(sentences)
    return [file,len(words),average_word_length,len(sentences),average_sentence_length]

inputs = read_all_files('input')
outputs = read_all_files('output')
goldensources = read_all_files('goldensource')

csv_scores = []
goldensource_stats_csv = []
input_stats_csv = []
stats_cols = ["File", "Words","Avg. Word Len","Sentences","Avg. Sentence Len"]
score_cols = ["File","R1_Precision","R1_Recall","R1_Fmeasure","R2_Precision","R2_Recall","R2_Fmeasure","RL_Precision","RL_Recall","RL_Fmeasure","Words","Avg. Word Len","Sentences","Avg. Sentence Len"]

for goldensource in goldensources:
    print(goldensource)
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

for input in inputs:
    input_stats_csv.append(file_stats("input/",input))

export_to_csv(input_stats_csv,stats_cols,"input_stats.csv")
export_to_csv(csv_scores,score_cols,"eval.csv")
export_to_csv(goldensource_stats_csv,stats_cols,"goldensource_stats.csv")







