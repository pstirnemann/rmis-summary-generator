'''
Summary Functions:

This code contains different summary functions which implement different models and approaches. 
'''
from helper_functions import *
import os
from dotenv import load_dotenv
load_dotenv('/.env')
import nltk
from nltk.tokenize import sent_tokenize
import openai
import json
import torch
from transformers import BertTokenizerFast, EncoderDecoderModel, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Initialize openAPI 
openai.api_key = os.getenv("OPENAI_API_KEY")
engine_list = openai.Engine.list()

# -------------------------------------------
# Summary functions
# -------------------------------------------
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

# Summarize with German Model BERT
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = 'mrm8488/bert2bert_shared-german-finetuned-summarization'
tokenizer = BertTokenizerFast.from_pretrained(ckpt)
model = EncoderDecoderModel.from_pretrained(ckpt).to(device)
def generate_german_summary(text):
   inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
   input_ids = inputs.input_ids.to(device)
   attention_mask = inputs.attention_mask.to(device)
   output = model.generate(input_ids, attention_mask=attention_mask, min_length=150, max_length=200)
   return tokenizer.decode(output[0], skip_special_tokens=True)

# Translate Text to German then summarize with GPT3 and translate back
def translate_summarize(text,max_tokens):
    translation = translate(text, "EN")
    summary = gpt3_summarize(translation, max_tokens)
    res = translate(summary, "DE")
    return res

# 
def biomedical_summary(text):
    tokenizer = AutoTokenizer.from_pretrained("Blaise-g/longt5_tglobal_large_scitldr")
    model = AutoModelForSeq2SeqLM.from_pretrained("Blaise-g/longt5_tglobal_large_scitldr")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    eng_text = translate(text, "EN")
    summary = summarizer(eng_text)
    ger_text = translate(summary[0]['summary_text'], "DE")
    return ger_text
