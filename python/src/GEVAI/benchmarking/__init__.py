import os
import time


def write_to_file(file_path, text, write_mode='a'):
    try:
        with open(file_path, write_mode) as file:
            file.write(str(text))
    except Exception as e:
        print(f"An error occurred: {e}")


def init_file(new_file, ex_post_explainers):
    folders = ['results', 'models']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    if not os.path.exists(new_file):
        create_benchmark_file(ex_post_explainers, new_file)
    else:
        # df = pd.read_csv(new_file)
        # ex_post_columns = [col for col in df.columns if 'Ex post' in col.lower()]
        # if len(ex_post_columns) != len(ex_post_explainers):
        #     create_benchmark_file(ex_post_explainers, new_file, 'w')
        # else:

        # If last line is not finished, add new line to ensure next benchmark entry is written to file correctly
        with open(new_file, 'r') as file:
            if file.readlines()[-1].rstrip('\n').endswith(','):
                write_to_file(new_file, "\n")


def create_benchmark_file(ex_post_explainers, new_file, write_mode='a'):
    formatted_explainers = [f"Ex post ({explainer})" for explainer in ex_post_explainers]
    write_to_file(
        new_file,
        f"Load configuration,Load dataset,Ad hoc type,Get all ad hoc explainers,Ad hoc model,Hypothesis,{','.join(formatted_explainers)}\n",
        write_mode
    )


def time_function(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    return time.time() - start, result
