


def get_a_priori_explainer(type, conf):
    if type == "Configuration":
        from GEVAI.apriori.loading import configuration_loading
        return configuration_loading(conf)
    elif type == "PandasLoad":
        from GEVAI.apriori.loading import data_loading
        return data_loading(conf)
    return None