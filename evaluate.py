from pyrouge import Rouge155

# Evaluate using pyrouge
r = Rouge155()
r.system_dir = 'output'
r.model_dir = 'goldensource'
r.system_filename_pattern = 'summary.(\d+).txt'
r.model_filename_pattern = '.summary.#ID#.txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)