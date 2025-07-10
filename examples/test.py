# Example for implementation for check_url_availability/

from url_availability_checker import runner


website_txt_file_input = 'data/input/list_50_urls.txt'
path_to_output = 'data/output/'


runner.run_program(website_txt_file_input, path_to_output)


