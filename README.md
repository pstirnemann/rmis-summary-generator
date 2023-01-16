# rmis-summary-generator

## Recommendation: Using a virtual environment

How to create a virtual environment in pythion: (https://towardsdatascience.com/virtual-environments-104c62d48c54)

## Prerequisites:

### Requirements

All required packages are listed in the requirements.txt file. To install the requirements run the comand. 

`pip install -r requirements.txt`

### OpenAI

1. Go to https://openai.com to create an account
2. Create a personal API Key
3. Add a .env File
4. Add your API Key in the .evn file `OPENAI_API_KEY = YOUR_KEY`

## How to use: 

1. Add three folders to the project: 
- input
- output
- goldensource
2. Place your input files in the created input folder
3. Place a goldensourcefile (existing abstract) in the folder golden source
4. Adjust the different approaches to use your own input and goldensource files
- Example read a file `my_text = read_text(input/myFileName.txt)`
- Example save a file `save_summary(my_summary_variable, "filenName.txt")`
- Example evaluate a summary `my_evaluation = eval_rouge(my_generated_summary, my_goldensource)`
