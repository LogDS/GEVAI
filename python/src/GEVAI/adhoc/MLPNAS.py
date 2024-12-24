from torch.testing._internal.common_fsdp import MLP

from GEVAI.adhoc.generic_algorithm import GenericAlgorithm
from GEVAI.adhoc.mlpnas.mlpnas import MLPNAS

class MLPNAS_(GenericAlgorithm):
    def __init__(self, conf):
        self.conf = conf

    def __call__(self, *args, **kwargs):
        training_x, training_y = args[0], args[1]
        # training_x = list(df.keys())
        # training_y = [df(x) for x in training_x]
        nas_object = MLPNAS(training_x, training_y, self.conf)
        data = nas_object.search()
        sequences = list(map(lambda x: x[0], data))
        for sequence in sequences:
            one_model = MLPNAS(training_x, training_y, self.conf)
            # Recreating the architecture from the sequence that was dumped
            model = one_model.create_architecture(sequence)
            # Setting up the weights associated to the neural networks
            one_model.model_generator.set_model_weights(model)
            yield model

class MLPNAS_Load(GenericAlgorithm):
    def __init__(self, conf):
        self.conf = conf


    def __call__(self, *args, **kwargs):
        training_x, training_y = args[0], args[1]
        nas_object = MLPNAS(training_x, training_y, self.conf)
        from GEVAI.adhoc.mlpnas.utils import get_latest_folder
        nas_object.load_from_configuration_folder(get_latest_folder())
        N = nas_object.get_all_searches()
        for i in range(N):
            model = MLPNAS(training_x, training_y, self.conf)
            model.load_from_configuration_folder(get_latest_folder())
            yield model.load_ith_search_result(i)


