from helper_functions import *
from summary_functions import *
from dotenv import load_dotenv
load_dotenv('/.env')

# -------------------------------------------
# Execution
# -------------------------------------------

# Load inputfiles
input_files = read_all_files('input')

'''
# Approach 1 (GPT-3)
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
print("Start Translate Approach")
for input in input_files:
    text = read_text('input/' + input)
    summary = translate_summarize(text)
    save_summary(summary,input,'T')
    print(input + " : Done")
print("End Translate Approach")
    
#Development 