"""
Original source: https://github.com/codeaway23/MLPNAS/
"""

import math
import tensorflow.python.keras.backend as K
import shap
from keras.api.utils import to_categorical
from keras.api.utils import pad_sequences
import torch
from GEVAI.adhoc.mlpnas.controller import Controller
from GEVAI.adhoc.mlpnas.mlp_generator import MLPGenerator
from GEVAI.adhoc.mlpnas.utils import *

fun_dict = {"sigmoid": lambda x: "(1 / (1 + exp(-(" + x + "))))",
            "softmax": lambda x: "exp(" + x + ")",  ##/ sum(exp("+x+"))
            "softplus": lambda x: "log(exp(" + x + ") + 1)",
            "softsign": lambda x: x + " / (abs(" + x + ") + 1)",
            "tanh": lambda x: "((exp(" + x + ") - exp(-(" + x + "))) / (exp(" + x + ") + exp(-(" + x + "))))",
            "relu": lambda x: "max(0,(" + x + "))"}

import sympy

fun_dict_sympy = {"sigmoid": lambda x: 1.0 / (1.0 + sympy.exp(-x)),
                  "relu": lambda x: max(0.0, x),
                  "tanh": lambda x: (sympy.exp(x) - sympy.exp(-x)) / (sympy.exp(x) + sympy.exp(-x)),
                  "softsign": lambda x: x / (sympy.Abs(x) + 1),
                  "softplus": lambda x: sympy.log(sympy.exp(x) + 1),
                  "softmax": lambda x: sympy.exp(x)}


class MyNeuron:
    def __init__(self):
        self.bias = None


def explain(hypothesis):
    input_values = None
    for layerNum, layer in enumerate(hypothesis.layers):
        W = layer.get_weights()
        activation = layer.activation.__name__
        weights = W[0]
        biases = W[1]
        if input_values is None:
            input_values = list(map(lambda x: f'x{x}', range(len(weights))))

        nextLayerNNeuonMax = max(map(len, weights))
        output_neurons = [list() for _ in range(nextLayerNNeuonMax)]
        for fromNeuronNum, wgt in enumerate(weights):
            for toNeuronNum, wgt2 in enumerate(wgt):
                output_neurons[toNeuronNum].append((f'{input_values[fromNeuronNum]} * {wgt2}'))

        if layer.use_bias:
            for toNeuronNum, bias in enumerate(biases):
                output_neurons[toNeuronNum].append(f'{bias}')

        output_neurons = list(map(lambda x: fun_dict[activation](" + ".join(x)), output_neurons))
        if activation == "softmax":
            joint_sum = " + ".join(output_neurons)
            output_neurons = list(map(lambda x: f"({x})/({joint_sum})", output_neurons))
        input_values = output_neurons
        output_neurons = []

    return input_values


class MLPNAS(Controller):
    def __init__(self, x, y, conf):
        self.x = x
        self.y = y
        self.target_classes = conf.TARGET_CLASSES
        self.controller_sampling_epochs = conf.CONTROLLER_SAMPLING_EPOCHS
        self.samples_per_controller_epoch = conf.SAMPLES_PER_CONTROLLER_EPOCH
        self.controller_train_epochs = conf.CONTROLLER_TRAINING_EPOCHS
        self.architecture_train_epochs = conf.ARCHITECTURE_TRAINING_EPOCHS
        self.controller_loss_alpha = conf.CONTROLLER_LOSS_ALPHA
        self.data = []
        self.nas_data_log = 'LOGS/nas_data.pkl'
        self.val_accuracy = conf.TARGET_SCORE
        clean_log()
        super().__init__(conf)
        self.model_generator = MLPGenerator(conf)
        self.controller_batch_size = len(self.data)
        self.controller_input_shape = (1, conf.MAX_ARCHITECTURE_LENGTH - 1)
        if self.use_predictor:
            self.controller_model = self.hybrid_control_model(self.controller_input_shape, self.controller_batch_size)
        else:
            self.controller_model = self.control_model(self.controller_input_shape, self.controller_batch_size)

    def load_from_configuration_folder(self, folder_name):
        """
        Loads the configuration that was inferred at training time
        :param folder_name:     Folder where the configuration resides
        :return:                Loaded data that is now part defining the object
        """
        self.nas_data_log = os.path.join(folder_name, 'nas_data.pkl')
        with open(self.nas_data_log, 'rb') as f:
            self.data = pickle.load(f)
        self.model_generator.load_from_configuration_folder(folder_name)
        return self.data

    def create_architecture(self, sequence):
        if self.target_classes == 2:
            self.model_generator.loss_func = 'binary_crossentropy'
        model = self.model_generator.create_model(sequence, np.shape(self.x[0]))
        model = self.model_generator.compile_model(model)
        return model

    def train_architecture(self, model):
        x, y = unison_shuffled_copies(self.x, self.y)
        history = self.model_generator.train_model(model, x, y, self.architecture_train_epochs)
        return history

    def append_model_metrics(self, sequence, history, pred_accuracy=None):
        if len(history.history[self.val_accuracy]) == 1:
            if pred_accuracy:
                self.data.append([sequence,
                                  history.history[self.val_accuracy][0],
                                  pred_accuracy])
            else:
                self.data.append([sequence,
                                  history.history[self.val_accuracy][0]])
            print(self.val_accuracy + ': ', history.history[self.val_accuracy][0])
        else:
            val_acc = np.ma.average(history.history[self.val_accuracy],
                                    weights=np.arange(1, len(history.history[self.val_accuracy]) + 1),
                                    axis=-1)
            if pred_accuracy:
                self.data.append([sequence,
                                  val_acc,
                                  pred_accuracy])
            else:
                self.data.append([sequence,
                                  val_acc])
            print(self.val_accuracy + ': ', val_acc)

    def prepare_controller_data(self, sequences):
        controller_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        xc = controller_sequences[:, :-1].reshape(len(controller_sequences), 1, self.max_len - 1)
        yc = to_categorical(controller_sequences[:, -1], self.controller_classes)
        val_acc_target = [item[1] for item in self.data]
        return xc, yc, val_acc_target

    def get_discounted_reward(self, rewards):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        for t in range(len(rewards)):
            running_add = 0.
            exp = 0.
            for r in rewards[t:]:
                running_add += self.controller_loss_alpha ** exp * r
                exp += 1
            discounted_r[t] = running_add
        discounted_r = (discounted_r - discounted_r.mean()) / discounted_r.std()
        return discounted_r

    def custom_loss(self, target, output):
        baseline = 0.5
        reward = np.array([item[1] - baseline for item in self.data[-self.samples_per_controller_epoch:]]).reshape(
            self.samples_per_controller_epoch, 1)
        discounted_reward = self.get_discounted_reward(reward)
        loss = - K.log(output) * discounted_reward[:, None]
        return loss

    def train_controller(self, model, x, y, pred_accuracy=None):
        if self.use_predictor:
            self.train_hybrid_model(model,
                                    x,
                                    y,
                                    pred_accuracy,
                                    self.custom_loss,
                                    len(self.data),
                                    self.controller_train_epochs)
        else:
            self.train_control_model(model,
                                     x,
                                     y,
                                     self.custom_loss,
                                     len(self.data),
                                     self.controller_train_epochs)

    def get_all_searches(self):
        return len(self.data)

    def load_ith_search_result(self, i):
        # Retrieving the sequence information from the dump
        self.sequences = list(map(lambda x: x[0], self.data))
        sequence = self.sequences[i]
        print('Architecture: ', self.decode_sequence(sequence))
        # Recreating the architecture from the sequence that was dumped
        model = self.create_architecture(sequence)
        # Setting up the weights associated to the neural networks
        self.model_generator.set_model_weights(model)
        return model

    def load_search_result(self):
        # Retrieving the sequence information from the dump
        self.sequences = list(map(lambda x: x[0], self.data))
        for i, sequence in enumerate(self.sequences):
            print('Architecture: ', self.decode_sequence(sequence))
            # Recreating the architecture from the sequence that was dumped
            model = self.create_architecture(sequence)
            # Setting up the weights associated to the neural networks
            self.model_generator.set_model_weights(model)

    def explain_search_result(self, colnames, howMuchSample=.3, nsamples=500):
        torch.set_grad_enabled(True)
        nsamples = max(nsamples, len(colnames) + 10)
        # Retrieving the sequence information from the dump
        print("Feature Names: " + str(colnames))
        self.sequences = list(map(lambda x: x[0], self.data))
        for i, sequence in enumerate(self.sequences):
            print('Architecture: ', self.decode_sequence(sequence))
            # Recreating the architecture from the sequence that was dumped
            model = self.create_architecture(sequence)
            # Setting up the weights associated to the neural networks
            self.model_generator.set_model_weights(model)
            explain(model)

            howSample = min(math.ceil(howMuchSample * len(self.x)), len(self.x))
            explainer = shap.DeepExplainer(model, self.x[:howSample])
            shap_values = explainer.shap_values(self.x)
            shap.summary_plot(shap_values, self.x, max_display=10)  # .png,.pdf will also support here

    def make_testing_predictions(self, x, y):
        # Retrieving the sequence information from the dump
        models_result = []
        self.sequences = list(map(lambda x: x[0], self.data))
        for i, sequence in enumerate(self.sequences):
            for_current_model = []
            print('Architecture: ', self.decode_sequence(sequence))
            # Recreating the architecture from the sequence that was dumped
            model = self.create_architecture(sequence)
            # Setting up the weights associated to the neural networks
            self.model_generator.set_model_weights(model)
            for_current_model.append(list(zip(model.predict(x), y)))
            models_result.append(for_current_model)
        return models_result

    def search(self):
        for controller_epoch in range(self.controller_sampling_epochs):
            print('------------------------------------------------------------------')
            print('                       CONTROLLER EPOCH: {}'.format(controller_epoch))
            print('------------------------------------------------------------------')
            self.sequences = self.sample_architecture_sequences(self.controller_model,
                                                                self.samples_per_controller_epoch)
            if self.use_predictor:
                pred_accuracies = self.get_predicted_accuracies_hybrid_model(self.controller_model, self.sequences)
            for i, sequence in enumerate(self.sequences):
                print('Architecture: ', self.decode_sequence(sequence))
                model = self.create_architecture(sequence)
                history = self.train_architecture(model)
                if self.use_predictor:
                    self.append_model_metrics(sequence, history, pred_accuracies[i])
                else:
                    self.append_model_metrics(sequence, history)
                print('------------------------------------------------------')
            xc, yc, val_acc_target = self.prepare_controller_data(self.sequences)
            self.train_controller(
                self.controller_model,
                xc,
                yc,
                val_acc_target[-self.samples_per_controller_epoch:]
            )
        with open(self.nas_data_log, 'wb') as f:
            pickle.dump(self.data, f)
        log_event()
        return self.data
