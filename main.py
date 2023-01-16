import os
from dotenv import load_dotenv
load_dotenv('/.env')
import nltk
from nltk.tokenize import sent_tokenize
import openai
import json
from transformers import GPT2TokenizerFast, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5Model, T5ForConditionalGeneration
from rouge_score import rouge_scorer

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
engine_list = openai.Engine.list()


# Read Text from a .txt File
def read_text(path):
    fileObject = open(path, "r")
    return fileObject.read()

# Split Text into Tokens
def split_text(text, n):
    sentences = sent_tokenize(text)
    subtexts = []
    subtext = ""
    token_count = 0
    for sentence in sentences:
        token_count += len(nltk.word_tokenize(sentence))
        if token_count > n:
            subtexts.append(subtext)
            subtext = ""
            token_count = 0
        subtext += sentence + " "
    if subtext:
        subtexts.append(subtext)
    return subtexts

# Translate (optional)
#   content = Textinput to translate
#   languageKey = DE --> Translate to German | EN --> Translate to English
def translate(content,languageKey):
    if languageKey == "DE":
        input = "Translate to German: " + content
    else:
        input = "Translate to English: " + content

    result = openai.Completion.create(
        engine="text-davinci-003",
        prompt=input,
        temperature=0.3,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        #stop=["\n"]
        )
    json_object = json.loads(str(result))
    return json_object['choices'][0]['text']


# Summarize GPT-3
def gpt3_summarize(content, max_tokens):
    input = content + " Tl;dr"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=input,
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        #stop=["\n"]
        )
    json_object = json.loads(str(response))
    return json_object['choices'][0]['text']


# concatenate Text
def concatenate_text(subtexts):
    return " ".join(subtexts)

# Evaluate
def eval_rouge(input, goldensource):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(input,goldensource)

# Save Summary
def save_summary(summary, fileName):
    with open('output/'+fileName, 'w') as f:
        f.write(summary)
        f.close()


# Approach 1 (GPT-3)
text_1 = read_text("input/digitale_medien.txt")
goldensource_1 = read_text("input/digitale_medien_summary.txt")
def final_summary(text):
    res = []
    text_tokens = split_text(text, 700)
    for text in text_tokens:
        res.append(gpt3_summarize(text, 150))
    return concatenate_text(res)

summary_1_1 = final_summary(text_1)

evaluation_1_1 = eval_rouge(summary_1_1, goldensource_1)

save_summary(summary_1_1, "summary_1_1.txt")
print("Summary: " + summary_1_1)
print("----------------------")
print("Evaluation: " )
print(evaluation_1_1)

# Approach 2 (German Model)

# Approach 3 (English Medical Model)

# Approach 4 (Translation & GPT-3)
def translate_summarize(text):
    text_token = split_text(text, 700)
    translation = []
    for text in text_token:
        translation.append(translate(text, "EN"))
    res = []
    for text in translation:
        res.append(gpt3_summarize(text, 150))
    translated_res = []
    for text in res:
        translated_res.append(translate(text, "DE"))
    return concatenate_text(translated_res)

#summary_4_1 = translate_summarize(text_1)
    

# Approach 5 (Translation & Medical Model)



#Development 


#Try



#Try Try
def summarize_german(input):
    tokenizer = AutoTokenizer.from_pretrained("Einmalumdiewelt/T5-Base_GNAD")
    model = AutoModelForSeq2SeqLM.from_pretrained("Einmalumdiewelt/T5-Base_GNAD")
    print(len(tokenizer.tokenize(input)))
    summarizer = pipeline(
        "summarization",
         model=model,
         tokenizer=tokenizer,
         min_length = 200, 
         max_length = 280)
    return summarizer(input)

def translate2(input):
    tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")
    input_ids = tokenizer(input, return_tensors="pt", add_special_tokens=False).input_ids
    output_ids = model.generate(input_ids)[0]
    print(tokenizer.decode(output_ids, skip_special_tokens=True))

def summarize_english(input):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    print(len(tokenizer.tokenize(input)))
    summarizer = pipeline(
        "summarization",
         model=model,
         tokenizer=tokenizer,
         min_length = 200, 
         max_length = 280)
    return summarizer(input)
    


