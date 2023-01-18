import os
# -------------------------------------------
# Helper functions
# -------------------------------------------

# Scan directory for input files
def read_all_files(directory):
    txt_files = []
    filenames = os.listdir(directory)
    for filename in filenames:
        if filename.endswith('.txt'):
            txt_files.append(filename)
    return txt_files

# Read Text from a .txt File
def read_text(path):
    fileObject = open(path, "r")
    return fileObject.read()

# Generate new filename to store outputfiles
def generate_new_name(file,approach):
    parts = file.split(".")
    new_name = parts[0] + "." + approach + "." + parts[1] + "." + parts[2]
    return new_name

# Save Summary
def save_summary(summary,file_name, approach_id):
    new_file_name = generate_new_name(file_name, approach_id)
    with open('output/'+new_file_name, 'w') as f:
        f.write(summary)
        f.close()

# concatenate Text
def concatenate_text(subtexts):
    return " ".join(subtexts)