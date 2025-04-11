import random
import sys
import pandas as pd
from GEVAI.adhoc.mlpnas.utils import *
from GEVAI.adhoc.mlpnas.mlpnas import MLPNAS
import yaml

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def data_loading(file, col, shuffle=False):
    data = pd.read_csv(file)
    xd = data.drop(col, axis=1, inplace=False)
    x = data.drop(col, axis=1, inplace=False).values
    y = None
    colN = xd.columns
    if conf.IS_TARGET_CATEGORICAL:
        y = pd.get_dummies(data[col]).values
    else:
        y = data[col].values
    if shuffle:
        l = list(range(0,len(data)))
        random.shuffle(l)
        return (x[l], y[l], colN)
    else:
        return (x,y,colN)

if __name__ == "__main__":
    file_conf = "black_box_parameters.yaml"
    action = "training" # "load", "training", "testing", "explain"
    if len(sys.argv) > 1:
        file_conf = sys.argv[1]
    if len(sys.argv) > 2:
        action = sys.argv[2]
    conf = None
    with open(file_conf, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
            conf = Struct(**conf)
        except yaml.YAMLError as exc:
            sys.exit(1)
    if action == "training":
        """
        Training the networks and dumping those in the logs folder
        """
        x,y,ignore = data_loading(conf.CSV_TRAINING, conf.CLASS_COLUMN)
        nas_object = MLPNAS(x, y, conf)
        data = nas_object.search()
    elif action == "plot_training":
        """
        Doing some preliminary plotting
        """
        get_top_n_architectures(conf.TOP_N, conf.TARGET_CLASSES, conf.nodes, conf.activation_functions)
        # get_accuracy_distribution()
    elif action == "load":
        nas_object = MLPNAS(None, None, conf)
        nas_object.load_from_configuration_folder(get_latest_folder())
        nas_object.load_search_result()
    elif action == "testing":
        x,y,ignore = data_loading(conf.CSV_TRAINING, conf.CLASS_COLUMN)
        nas_object = MLPNAS(x, y, conf)
        nas_object.load_from_configuration_folder(get_latest_folder())
        training_mse = calculate_mse_per_architecture(nas_object.make_testing_predictions(x,y))
        x,y,ignore = data_loading(conf.CSV_TESTING, conf.CLASS_COLUMN)
        testing_mse = calculate_mse_per_architecture(nas_object.make_testing_predictions(x,y))
        print(list(zip(training_mse, testing_mse)))
    elif action == "explain":
        x,y,colnames = data_loading(conf.CSV_TRAINING, conf.CLASS_COLUMN, True)
        nas_object = MLPNAS(x, y, conf)
        nas_object.load_from_configuration_folder(get_latest_folder())
        nas_object.explain_search_result(colnames, howMuchSample=.2, nsamples=10)



