# rmis-summary-generator

## Recommendation: Using a virtual environment

How to create a virtual environment in pythion: (https://towardsdatascience.com/virtual-environments-104c62d48c54)

## Prerequisites:

### Requirements

**The code is developed using Python 3.10.9**

All required packages are listed in the requirements.txt file. To install the requirements run the comand. 

`pip install -r requirements.txt`

### OpenAI API

The provided research prototype uses the GPT-3 language model provided by OpenAI's API.
in order to execute the programm an OpenAI API key is needed. 
Follow the steps below to generate and use a personal API key.

1. Go to https://openai.com to create an account
2. Create a personal API Key
3. Add a .env File
4. Add your API Key in the .evn file `OPENAI_API_KEY = YOUR_KEY`

## What the project contains:

### main.py

The file contains the main code to summarize each text contained in the input folder and saving the generated summary in the output folder.
Multiple approaches are developed and all of them will run once the main.py script is executed. 
Durring the execution the script will write out some logs about the current task to the console. 

==The script uses an API provided by openAI which generatescosts. Be carefull how to use it. 
To avoid costs the following approaches need to be commented out: 
    - Approach 1 (GPT-3)
    - Approach 4 (Translation & GPT-3)==

### evaluate.py

This file contains the code to generate the Evaluation Files. 
For the generated summaries the Rouge1, Rouge2 and RougeL Score is calculated using rouge_score
Additionally some statistics about Words/Sentences for the input, goldensource and summaries are added
The script generates 3 outputfiles
   - eval.csv: Rougescores and statistics for the summaries
   - input_stats.csv: Statistics for the input files
   - goldensource_stats.csv: Statistics for the goldensource files

### summary_functions.py

This code contains different summary functions which implement different models and approaches. 

### helper_functions.py

This file contains various helper functions used in the different scripts. 

## Run

Before running the code the following actions are needed: 

1. Add a folder called "output"
2. Place your input files int the "input" folder
    * Required filename format: filename.ID.txt
3. Place the reference summaries in the "goldensource" folder
    * Required filename format: filename.ID.txt  (Corresponding input file ID)
4. Run main.py
    * The script will generate summaries using the described approaches and save them in the output folder. 
    * Output Filename format: filename.APPROACH_ID.ID.txt (APPROACH_ID = Approach used to summarize, ID = Corresponding input file ID)
5. Run evaluate.py
    * The script will generate the 3 described CSV-Files
