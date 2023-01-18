from rouge_score import rouge_scorer
from helper_functions import *
import pandas as pd

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def contains_substring(string, substring):
    return substring in string

def export_to_csv(list):
    df = pd.DataFrame(list,columns=["File","R1_Precision","R1_Recall","R1_Fmeasure","RL_Precision","RL_Recall","RL_Fmeasure"])
    df.to_csv("eval.csv",index=False,encoding='utf-8')

outputs = read_all_files('output')
goldensources = read_all_files('goldensource')

csv_output = []
for goldensource in goldensources:
    golden_text = read_text('goldensource/'+goldensource)
    id = goldensource.split(".")[1]
    predictions = list(filter(lambda x: contains_substring(x,id),outputs))
    for prediction in predictions:
        prediction_text = read_text('output/'+prediction)
        scores = scorer.score(golden_text, prediction_text)
        rouge1 = scores.get("rouge1")
        rougeL = scores.get("rougeL")
        csv_output.append([prediction,rouge1.precision,rouge1.recall,rouge1.fmeasure,rougeL.precision,rougeL.recall,rougeL.fmeasure])

export_to_csv(csv_output)