from GEVAI.expost.ExPost import ExPost
from GEVAI.utils import fullname

fun_dict = {"sigmoid": lambda x: "(1 / (1 + exp(-("+x+"))))",
            "softmax": lambda x: "exp("+x+")",##/ sum(exp("+x+"))
            "softplus": lambda x: "log(exp("+x+") + 1)",
            "softsign": lambda x: x+" / (abs("+x+") + 1)",
            "tanh": lambda x: "((exp("+x+") - exp(-("+x+"))) / (exp("+x+") + exp(-("+x+"))))",
            "relu": lambda x: "max(0,("+x+"))"}


class KerasExplainString(ExPost):

    def __init__(self, conf):
        self.conf = conf

    def acceptingType(self):
        import keras.engine.sequential
        return keras.engine.sequential.Sequential

    def __call__(self, *args, **kwargs):
        """
        Given a keras sequential model as an input, this function
        expresses the neural network as a list of equations
        """
        hypothesis = args[0]
        ## TODO: do something depending on the type!
        input_values = None
        h = fullname(hypothesis)
        if h == 'keras.engine.sequential.Sequential':
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
            return [input_values]
        else:
            print("Unsupported BlackBox explainer")
            return []