# Example for implementation for check_url_availability/

from url_availability_checker import runner

'''import os
print("Current Working Directory:", os.getcwd())
'''

website_txt_file_input = 'data/input/list_50_urls.txt'
path_to_output = 'data/output/'
jsonl_output = True
csv_output = True
batch_size = 260
workers = 5
save_incrementally = False

runner.run_program(website_txt_file_input, path_to_output, jsonl_output, csv_output, batch_size, save_incrementally, workers)


