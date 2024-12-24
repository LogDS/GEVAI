

def get_ad_hoc_explainer(eplainer_type, conf):
    trainer = None
    if eplainer_type == "MLPNAS":
        from GEVAI.adhoc.MLPNAS import MLPNAS_
        trainer = MLPNAS_(conf)
    elif eplainer_type == "MLPNAS_pretrained":
        from GEVAI.adhoc.MLPNAS import MLPNAS_Load
        trainer = MLPNAS_Load(conf)
    elif eplainer_type == "GenericAlgorithm": ## TODO: use the configuration parameters to actually instantiate the algorighm
        from GEVAI.adhoc.generic_algorithm import GenericAlgorithm
        trainer = GenericAlgorithm()
    return trainer