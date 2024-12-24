
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

import yaml
import sys

def configuration_loading(file_conf):
    with open(file_conf, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
            conf = Struct(**conf)
            return conf
        except yaml.YAMLError as exc:
            return None

import pandas as pd
import random

def data_loading(conf):
    file, col, shuffle = conf.CSV_TRAINING, conf.CLASS_COLUMN, conf.DATA_SHUFFLE
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