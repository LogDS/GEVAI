def get_a_priori_explainer(type, conf, explainer_type=None):
    if type == "Configuration":
        from GEVAI.apriori.loading import configuration_loading
        return configuration_loading(conf)
    elif type == "PandasLoad":
        from GEVAI.apriori.loading import data_loading
        return data_loading(conf, explainer_type)
    return None
