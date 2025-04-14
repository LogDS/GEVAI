import os
import time

from GEVAI import (get_ad_hoc_explainer,
                   get_a_priori_explainer,
                   get_ex_post_explainer,
                   benchmarking)
from GEVAI.benchmarking import time_function

def get_explanation_and_explainer(ex_post_expl, config):
    explainer = get_ex_post_explainer(ex_post_expl, config)
    explanation = explainer(model, training_x=df[0])
    return explainer, explanation

if __name__ == '__main__':
    os.chdir('../')

    benchmarking_file_path = os.path.join("benchmark.csv")
    benchmarking.init_file(benchmarking_file_path)  # Load configuration,Load dataset,Ad hoc type,Get all ad hoc explainers,Load ad hoc explainers,Ad hoc model,Ex post (BlackBox),Ex post (ModelAgnostic)

    conf = "/home/giacomo/PyCharmProjects/GEVAI/black_box_parameters.yaml"
    ## Loading the configuration for the entire architecture
    start_time = time.time()
    conf = get_a_priori_explainer("Configuration", conf)
    end_time = time.time()
    loading_config_time = end_time - start_time

    ## Loading the dataset
    ## TODO: use the actual representation hierarchy
    start_time = time.time()
    df = get_a_priori_explainer("PandasLoad", conf)
    end_time = time.time()
    loading_dataset_time = end_time - start_time

    ## Use "MLPNAS_pretrained" if you already run the model over the dataset, so to use the persisted outcome
    ## of the model being trained, ad MLPNAS if you need to train this for the first time
    start_time = time.time()
    explainer_type = "DecisionTree"  # MLPNAS / MLPNAS_pretrained / GenericAlgorithm
    ad_hoc = get_ad_hoc_explainer(explainer_type, conf)
    models = list(ad_hoc(*df))
    end_time = time.time()

    generating_ad_hoc_time, loading_ad_hoc_time = None, None
    if explainer_type.endswith("pretrained"):
        loading_ad_hoc_time = end_time - start_time
    else:
        generating_ad_hoc_time = end_time - start_time

    ex_post_explainers = ["BlackBoxExplainer", "WhiteBoxExplainer", "LIME", "Shapely"]
    for model in models[:2]:
        ## TODO. Oliver, not all implemetations of Ad Hoc have a name. If no name attribute is available, you can use explainer_type
        # benchmarking.write_to_file(benchmarking_file_path, f"{loading_config_time},{loading_dataset_time},{explainer_type},{generating_ad_hoc_time},{loading_ad_hoc_time},{model.name},")
        loading_config_time, loading_dataset_time, generating_ad_hoc_time, loading_ad_hoc_time = None, None, None, None

        for ex_post_explainer in ex_post_explainers:
            ex_post_time, (ex_post, explanation) = time_function(
                get_explanation_and_explainer,
                ex_post_explainer,
                conf
            )
            benchmarking.write_to_file(benchmarking_file_path, f"{ex_post_time}," if ex_post_explainer != ex_post_explainers[-1] else f"{ex_post_time}\n")
