import os
import time


def write_to_file(file_path, text):
    try:
        with open(file_path, 'a') as file:
            file.write(str(text))
    except Exception as e:
        print(f"An error occurred: {e}")

def init_file(benchmarking_file):
    if not os.path.exists(benchmarking_file):
        write_to_file(benchmarking_file, "Load configuration,Load dataset,Ad hoc type,Get all ad hoc explainers,Load ad hoc explainers,Ad hoc model,Ex post (BlackBox),Ex post (Shapely),Ex post (LIME)\n")
    else:
        # If last line is not finished, add new line to ensure next benchmark is written to file correctly
        with open(benchmarking_file, 'r') as file:
            if file.readlines()[-1].rstrip('\n').endswith(','):
                write_to_file(benchmarking_file, "\n")

def time_function(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    return time.time() - start, result