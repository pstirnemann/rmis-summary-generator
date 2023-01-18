from helper_functions import *
from summary_functions import *
from dotenv import load_dotenv
load_dotenv('/.env')

# -------------------------------------------
# Execution
# -------------------------------------------

# Load inputfiles
input_files = read_all_files('input')


# Approach 1 (GPT-3)
def final_gpt3_summary(text):
    res = []
    text_tokens = split_text(text, 700)
    for text in text_tokens:
        res.append(gpt3_summarize(text, 150))
    return concatenate_text(res)

for input in input_files:
    text = read_text('input/' + input)
    summary = final_gpt3_summary(text)
    save_summary(summary,input,'GPT3')


#Commented out to save OpenAI Credits

#summary_1_1 = final_summary(text_1)

#evaluation_1_1 = eval_rouge(summary_1_1, goldensource_1)

#save_summary(summary_1_1, "summary_1_1.txt")
#print("Summary: " + summary_1_1)
#print("----------------------")
#print("Evaluation: " )
#print(evaluation_1_1)

# Approach 2 (German/Multilingual Model)

for input in input_files:
    text = read_text('input/' + input)
    summary = generate_german_summary(text)
    save_summary(summary,input,'GM')

#summary_2_1 = generate_german_summary(text_1)
#evaluation_2_1 = eval_rouge(summary_2_1, goldensource_1)

#save_summary(summary_2_1, "summary_2_1.txt")
#print("Summary: " + summary_2_1)
#print("----------------------")
#print("Evaluation: " )
#print(evaluation_2_1)

# Approach 3 (English Medical Model)



#print(generate_bio_summary(text_1))

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

#Commented out to save OpenAI Credits
#summary_4_1 = translate_summarize(text_1)
    

# Approach 5 (Translation & Medical Model)



#Development 