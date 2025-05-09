import os
import shutil
import pickle
import numpy as np
from itertools import groupby
from matplotlib import pyplot as plt
# from CONSTANTS import *
# from mlp_generator import MLPSearchSpace

########################################################
#                   DATA PROCESSING                    #
########################################################


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


########################################################
#                       LOGGING                        #
########################################################


def clean_log():
    log_directory = 'LOGS/'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    filelist = os.listdir('LOGS/')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            os.remove('LOGS/{}'.format(file))


def log_event():
    dest = 'LOGS'
    while os.path.exists(dest):
        dest = 'LOGS/event{}'.format(np.random.randint(10000))
    os.mkdir(dest)
    filelist = os.listdir('LOGS')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            shutil.move('LOGS/{}'.format(file),dest)


def get_latest_event_id():
    all_subdirs = ['LOGS/' + d for d in os.listdir('LOGS') if os.path.isdir('LOGS/' + d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return int(latest_subdir.replace('LOGS/event', ''))


########################################################
#                 RESULTS PROCESSING                   #
########################################################

def get_latest_folder():
    event = get_latest_event_id()
    return 'LOGS/event{}/'.format(event)

def load_nas_data():
    event = get_latest_event_id()
    data_file = 'LOGS/event{}/nas_data.pkl'.format(event)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data

def load_shared_weights():
    event = get_latest_event_id()
    data_file = 'LOGS/event{}/shared_weights.pkl'.format(event)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data

def sort_search_data(nas_data):
    val_accs = [item[1] for item in nas_data]
    sorted_idx = np.argsort(val_accs)[::-1]
    nas_data = [nas_data[x] for x in sorted_idx]
    return nas_data

########################################################
#                EVALUATION AND PLOTS                  #
########################################################

def get_top_n_architectures(n,tc,nodes,af):
    data = load_nas_data()
    weights = load_shared_weights()
    data = sort_search_data(data)
    from GEVAI.adhoc.mlpnas.mlp_generator import MLPSearchSpace
    search_space = MLPSearchSpace(tc,nodes,af)
    print('Top {} Architectures:'.format(n))
    for seq_data in data[:n]:
        print('Architecture', search_space.decode_sequence(seq_data[0]))
        print('Validation Accuracy:', seq_data[1])


def get_nas_accuracy_plot():
    data = load_nas_data()
    accuracies = [x[1] for x in data]
    plt.plot(np.arange(len(data)), accuracies)
    plt.show()


def get_accuracy_distribution():
    event = get_latest_event_id()
    data = load_nas_data()
    accuracies = [x[1]*100. for x in data]
    accuracies = [int(x) for x in accuracies]
    sorted_accs = np.sort(accuracies)
    count_dict = {k: len(list(v)) for k, v in groupby(sorted_accs)}
    plt.bar(list(count_dict.keys()), list(count_dict.values()))
    plt.show()

def calculate_mse_per_architecture(lsResults):
    mseList = []
    for x in lsResults:
        assert len(x)==1
        e = 0.0
        S = float(len(x[0]))
        for (ls, val) in x[0]:
            e += (ls[0]-val)**2.0
        mseList.append(e/S)
    return mseList

