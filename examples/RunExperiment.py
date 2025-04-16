import os
import time

from GEVAI import get_a_priori_explainer, get_ex_post_explainer, get_ad_hoc_explainer
from GEVAI.benchmarking import write_to_file, init_file, time_function


class RunExperiment:
    def __init__(self):
        self.conf = "black_box_parameters.yaml"
        self.explainer_types = [
            # Use "_pretrained" to avoid retraining
            "MLPNAS", "DecisionTree", "RipperK"
            # "GenericAlgorithm"
        ]
        self.ex_post_explainers = [
            "Metrics", "BlackBoxExplainer", "WhiteBoxExplainer", "LIME", "Shapely"
        ]
        self.benchmarking_file_path = os.path.join(
            f"results/benchmark_{len(self.ex_post_explainers)}ex_post_explainers.csv")
        self.df = None

    def get_explanation_and_explainer(self, ex_post_expl, config, model):
        explainer = get_ex_post_explainer(ex_post_expl, config)
        explanation = explainer(model, training_x=self.df[0], training_y=self.df[1])
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
        for model in models[:self.conf.TARGET_CLASSES]:
            from GEVAI.utils import fullname
            result_path = f'results/{fullname(model)}'
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            write_to_file(self.benchmarking_file_path,
                          f"{model.name if hasattr(model, 'name') else explainer_type},'{str(model)}',"
                          if model == models[
                              0] else f"{None},{None},{explainer_type},{None},{model.name if hasattr(model, 'name') else explainer_type},'{str(model)}',")

            if not self.conf.RUN_ALL_EX_POST:
                self.ex_post_explainers = ["Shapely"]

            for ex_post_explainer in self.ex_post_explainers:
                ex_post_time, (ex_post, explanation) = time_function(
                    self.get_explanation_and_explainer,
                    ex_post_explainer,
                    self.conf,
                    model
                )

                # Do not report time if no training was performed
                if not explanation or explanation is None or (isinstance(explanation, list) and len(explanation) == 0):
                    ex_post_time = None

                # Write training time and append new line if last ex post explainer
                write_to_file(
                    self.benchmarking_file_path,
                    f"{ex_post_time}," if ex_post_explainer != self.ex_post_explainers[-1] else f"{ex_post_time}\n"
                )

    def start_experiment(self):
        os.chdir('../')
        init_file(self.benchmarking_file_path, self.ex_post_explainers)

        # Loading the configuration for the entire architecture
        start_time = time.time()
        self.conf = get_a_priori_explainer("Configuration", self.conf)
        end_time = time.time()
        loading_config_time = end_time - start_time

        # Config determines whether to loop over every ad hoc explainer
        # Add "_pretrained" at end of type to avoid re-training
        if not self.conf.RUN_ALL_EXPLAINERS:
            self.explainer_types = ["MLPNAS"]

        for explainer_type in self.explainer_types:
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
