import os
import time

from GEVAI import get_a_priori_explainer, get_ex_post_explainer, get_ad_hoc_explainer
from GEVAI.benchmarking import write_to_file, init_file, time_function


class RunExperiment:
    def __init__(self):
        self.conf = None
        self.conf_dir = "black_box_parameters.yaml"
        self.explainer_types = [
            # Use "_pretrained" to avoid retraining
            "DecisionTree_pretrained"
            # "GenericAlgorithm"
        ]
        self.ex_post_explainers = [
            "WhiteBoxExplainer"
        ]
        self.benchmarking_file_path = os.path.join(
            f"results/benchmark_{len(self.ex_post_explainers)}ex_post_explainers.csv")
        self.results_path = None
        self.df = None

    def get_explanation_and_explainer(self, ex_post_expl, model):
        explainer = get_ex_post_explainer(ex_post_expl, self.conf)
        explanation = explainer(model, training_x=self.df[0], training_y=self.df[1], results_path=self.results_path)
        return explainer, explanation

    def run_ad_hoc(self, explainer_type):
        start_time = time.time()
        ad_hoc = get_ad_hoc_explainer(explainer_type, self.conf)
        models = list(ad_hoc(*self.df))
        end_time = time.time()
        generating_ad_hoc_time = end_time - start_time
        write_to_file(self.benchmarking_file_path, f"{explainer_type},{generating_ad_hoc_time},")
        return models

    def run_ex_post(self, models, explainer_type):
        for idx, model in enumerate(models[:self.conf.TARGET_CLASSES]):
            model_name = f"{model.name if hasattr(model, 'name') else explainer_type}"
            hypothesis = f"{str(model).replace(',',';')}"

            self.results_path = f'results/{model_name}_{idx}'
            if not os.path.exists(self.results_path):
                os.makedirs(self.results_path)

            write_to_file(
                self.benchmarking_file_path,
                f'{model_name},{hypothesis},'
                if model == models[0] else f'{None},{None},{explainer_type},{None},{model_name},{hypothesis},'
            )

            if not self.conf.RUN_ALL_EX_POST:
                self.ex_post_explainers = ["SHAP"]

            for ex_post_explainer in self.ex_post_explainers:
                if not (ex_post_explainer == 'SHAP' and explainer_type == 'RipperK'):  # TODO: Change this to computation timeout
                    ex_post_time, (ex_post, explanation) = time_function(
                        self.get_explanation_and_explainer,
                        ex_post_explainer,
                        model
                    )

                    # Do not report time if no training was performed
                    if not explanation or explanation is None or (isinstance(explanation, list) and len(explanation) == 0):
                        ex_post_time = None
                else:
                    ex_post_time = None

                # Write training time and append new line if last ex post explainer
                write_to_file(
                    self.benchmarking_file_path,
                    f"{ex_post_time}," if ex_post_explainer != self.ex_post_explainers[-1] else f"{ex_post_time}\n"
                )

    def start_experiment(self):
        os.chdir('../')
        init_file(self.benchmarking_file_path, self.ex_post_explainers)
        should_reload_config = True
        loop_count = 1

        for i in range(loop_count):
            for idx, explainer_type in enumerate(self.explainer_types):
                if should_reload_config or idx == 0:
                    self.conf = None
                    # Loading the configuration for the entire architecture
                    start_time = time.time()
                    self.conf = get_a_priori_explainer("Configuration", self.conf_dir)
                    end_time = time.time()
                    loading_config_time = end_time - start_time

                # Config determines whether to loop over every ad hoc explainer
                # Add "_pretrained" at end of type to avoid re-training
                if not self.conf.RUN_ALL_EXPLAINERS:
                    self.explainer_types = ["MLPNAS"]

                # Loading the dataset
                # TODO: use the actual representation hierarchy
                start_time = time.time()
                self.df = get_a_priori_explainer("PandasLoad", self.conf, explainer_type)
                end_time = time.time()
                loading_dataset_time = end_time - start_time

                write_to_file(self.benchmarking_file_path, f"{loading_config_time},{loading_dataset_time},")
                loading_config_time = None

                models = self.run_ad_hoc(explainer_type)
                self.run_ex_post(models, explainer_type)


if __name__ == '__main__':
    RunExperiment().start_experiment()
