

def get_ad_hoc_explainer(explainer_type, conf):
    trainer = None
    if explainer_type == "MLPNAS":
        from GEVAI.adhoc.MLPNAS import MLPNAS_
        trainer = MLPNAS_(conf)
    elif explainer_type == "MLPNAS_pretrained":
        from GEVAI.adhoc.MLPNAS import MLPNAS_Load
        trainer = MLPNAS_Load(conf)
    elif explainer_type == "GenericAlgorithm": ## TODO: use the configuration parameters to actually instantiate the algorithm
        from GEVAI.adhoc.generic_algorithm import GenericAlgorithm
        trainer = GenericAlgorithm()
    return trainer