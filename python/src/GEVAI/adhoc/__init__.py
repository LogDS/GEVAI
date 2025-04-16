import os
import pickle


def get_ad_hoc_explainer(explainer_type, conf):
    trainer = None
    if explainer_type.startswith("DecisionTree"):
        from GEVAI.adhoc.DecisionTree import DecisionTree_
        trainer = DecisionTree_(conf, explainer_type.endswith("pretrained"))
    elif explainer_type.startswith("RipperK"):
        from GEVAI.adhoc.RipperK import RipperK_
        trainer = RipperK_(conf, explainer_type.endswith("pretrained"))
    elif explainer_type == "MLPNAS":
        from GEVAI.adhoc.MLPNAS import MLPNAS_
        trainer = MLPNAS_(conf)
    elif explainer_type == "MLPNAS_pretrained":
        from GEVAI.adhoc.MLPNAS import MLPNAS_Load
        trainer = MLPNAS_Load(conf)
    elif explainer_type == "GenericAlgorithm":  ## TODO: use the configuration parameters to actually instantiate the algorithm
        from GEVAI.adhoc.generic_algorithm import GenericAlgorithm
        trainer = GenericAlgorithm()
    return trainer

def save_model(models, model_name):
    file_path = f"models/{model_name}_model.pkl"

    try:
        with open(file_path, 'wb') as f:
            pickle.dump(models, f)
        print(f"Trained {model_name} model saved to {file_path}")
    except Exception as e:
        print(f"Error saving models: {e}")

def load_model(model_name, should_load):
    file_path = f"models/{model_name}_model.pkl"

    if not should_load or not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"{model_name} model loaded from {file_path}")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None