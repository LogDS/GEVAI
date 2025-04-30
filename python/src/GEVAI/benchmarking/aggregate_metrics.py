import json
import csv
import os
from collections import defaultdict


def create_metrics_csv(root_folder_path, output_csv_file):
    grouped_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_prefixes = set()

    for dirpath, dirnames, filenames in os.walk(root_folder_path):
        for filename in filenames:
            if filename.startswith("metrics") and filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(dirpath, root_folder_path)
                full_name = os.path.join(relative_path, os.path.splitext(filename)[0])

                if '_' in full_name:
                    prefix = full_name.split('_')[0]
                    if 'sequential' in prefix:
                        prefix = "MLPNAS"
                else:
                    prefix = full_name
                all_prefixes.add(prefix)

                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for metric_name, values in data.items():
                        metric_name = metric_name.title().replace("_", " ")
                        if isinstance(values, dict):
                            for average_type, value in values.items():
                                grouped_data[prefix][metric_name][average_type.title()].append(value)
                        else:
                            grouped_data[prefix][metric_name]["--"].append(values)

    averaged_data = {}
    for prefix, metrics in grouped_data.items():
        averaged_data[prefix] = {}
        for metric_name, average_types in metrics.items():
            averaged_data[prefix][metric_name] = {}
            for avg_type, value_list in average_types.items():
                if value_list:
                    averaged_data[prefix][metric_name][avg_type] = round(sum(value_list) / len(value_list), 4)
                else:
                    averaged_data[prefix][metric_name][avg_type] = ""

    csv_rows = []
    sorted_prefixes = sorted(list(all_prefixes))
    header = ["Metric Type", "Average"] + sorted_prefixes
    csv_rows.append(header)

    metric_order = ["Accuracy", "F1 Score", "Precision", "Recall"]  # Enforce specific order
    average_order = ["--", "Macro", "Weighted"]

    for metric in metric_order:
        for avg in average_order:
            row = [metric, avg]
            for prefix in sorted_prefixes:
                if prefix in averaged_data and metric in averaged_data[prefix] and avg in averaged_data[prefix][metric]:
                    row.append(averaged_data[prefix][metric][avg])
                else:
                    row.append("")
            if any(row[2:]):  # Only add if there's data for this metric/average combination
                csv_rows.append(row)

    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)

    print(f"CSV file '{output_csv_file}' has been created successfully from '{root_folder_path}' and its subfolders.")


def csv_to_tabularray(csv_filepath, caption=None, label=None):
    latex_code = ""
    with open(csv_filepath, 'r', newline='') as csvfile:
        reader = list(csv.reader(csvfile))
        header = reader[0]
        data_rows = reader[1:]
        num_cols = len(header)

        latex_code += "\\definecolor{Gallery}{rgb}{0.925,0.925,0.925}\n"
        latex_code += "\\begin{table}[h!]\n"
        latex_code += "\\centering\n"
        latex_code += "\\begin{tblr}{\n"
        latex_code += f"  cells = {{c}},\n"
        latex_code += f"  row{{1}} = {{Gallery}},\n"

        # Determine rowspan for the first column
        row_spans = {}
        current_metric = None
        count = 0
        for i, row in enumerate(data_rows):
            metric = row[0]
            if metric == current_metric:
                count += 1
            else:
                if current_metric is not None or metric == "Average":
                    if count > 0:
                        start_row = 2 + i - count
                        row_spans[start_row] = count
                current_metric = metric
                count = 1
        if current_metric is not None and count > 1:
            start_row = 2 + len(data_rows) - count
            row_spans[start_row] = count

        for i in range(len(data_rows)):
            if i > 0 and data_rows[i][0] == data_rows[i - 1][0]:
                continue  # Skip rows that will be part of a rowspan

            row_index_latex = i + 2
            if row_index_latex in row_spans:
                latex_code += f"  cell{{{row_index_latex}}}{{1}} = {{r={row_spans[row_index_latex]}}}{{Gallery}},\n"
            elif i % 3 == 1 or data_rows[i][0] != 'Accuracy':
                latex_code += f"  row{{{row_index_latex}}} = {{Gallery}},\n"

        latex_code += f"  vlines,\n"
        latex_code += f"  hline{{1-3,5,7,9}} = {{-}}{{}},\n"
        latex_code += f"  hline{{4-9}} = {{2-5}}{{}},\n"
        latex_code += "}\n"

        # Header row
        latex_code += "  \\textbf{Metric Type} & \\textbf{Average} & "
        latex_code += " & ".join([f"\\textbf{{{h}}}" for h in header[2:]]) + " \\\\\n"

        # Data rows
        for i, row in enumerate(data_rows):
            if i > 0 and row[0] == data_rows[i - 1][0]:
                latex_code += "  & "  # Empty cell for the merged Metric Type
            else:
                latex_code += f"  \\textbf{{{row[0]}}} & "
            latex_code += f"  {row[1]} & "
            for j, cell in enumerate(row[2:]):
                try:
                    value = float(cell)
                    max_val_row = max([float(x) for x in row[2:] if x.replace('.', '', 1).isdigit()] or [None])
                    min_val_row = min([float(x) for x in row[2:] if x.replace('.', '', 1).isdigit()] or [None])
                    formatted_cell = f"{value:.4f}"
                    if max_val_row is not None and abs(value - max_val_row) < 1e-9:
                        formatted_cell = f"\\textbf{{\\color{{blue}}{{{formatted_cell}}}}}"
                    elif min_val_row is not None and abs(value - min_val_row) < 1e-9:
                        formatted_cell = f"\\color{{red}}{{{formatted_cell}}}"
                    latex_code += formatted_cell
                except ValueError:
                    latex_code += cell
                if j < len(row[2:]) - 1:
                    latex_code += " & "
            latex_code += " \\\\\n"

        latex_code += "\\end{tblr}\n"

        if caption:
            latex_code += f"\\caption{{{caption}}}\n"
        if label:
            latex_code += f"\\label{{{label}}}\n"

        latex_code += "\\end{table}\n"

    return latex_code


if __name__ == '__main__':
    folder_path = '../../../../results/test'
    output_csv_file = 'output.csv'
    create_metrics_csv(folder_path, output_csv_file)

    csv_file = 'output.csv'
    latex_table = csv_to_tabularray(
        csv_file,
        caption="Classification metrics for different ad hoc pipelines within GEVAI framework.",
        label="table:metrics"
    )
    print(latex_table)

    with open('table.tex', 'w') as f:
        f.write(latex_table)

    print("Table saved to 'table.tex'")
