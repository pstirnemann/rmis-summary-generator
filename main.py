'''
Main script:

This file contains the code to summarize each text in a folder and saving the generated summary.
There are multiple approaches which will run. 
Durring the execution the script writes out to the console about the current tasks
The script uses an API provided by openAI which generates costs. Be carefull how to use it. 
To avoid costs the following approaches need to be commented out: 
    - Approach 1 (GPT-3)
    - Approach 4 (Translation & GPT-3)
'''
from helper_functions import *
from summary_functions import *


# Load inputfiles
input_files = read_all_files('input')


# Approach 1 (GPT-3)

# Function to process the gpt3 summary where the text needs to be splitted before calling the API
'''
def final_gpt3_summary(text):
    res = []
    text_tokens = split_text(text, 700)
    for text in text_tokens:
        res.append(gpt3_summarize(text, 150))
    return concatenate_text(res)

print("Start GPT3 Approach")
for input in input_files:
    text = read_text('input/' + input)
    summary = final_gpt3_summary(text)
    save_summary(summary,input,'GPT3')
    print(input + " : Done")
print("End GPT3 Approach")

# Approach 2 (German/Multilingual Model)
print("Start GM Approach")
for input in input_files:
    text = read_text('input/' + input)
    summary = generate_german_summary(text)
    save_summary(summary,input,'GM')
    print(input + " : Done")
print("End GM Approach")

'''

# Approach 3 (Translation & Medical Model))



# Approach 4 (Translation & GPT-3)

def final_translate_summary(text):
    res = []
    text_tokens = split_text(text, 700)
    for text in text_tokens:
        res.append(translate_summarize(text, 150))
    return concatenate_text(res)

print("Start Translate Approach")
for input in input_files:
    text = read_text('input/' + input)
    summary = final_translate_summary(text)
    save_summary(summary,input,'T2')
    print(input + " : Done")
print("End Translate Approach")
    
#Development 