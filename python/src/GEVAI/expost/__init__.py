def get_ex_post_explainer(explainer_type, conf):
    if explainer_type == "BlackBoxExplainer":
        from GEVAI.expost.KerasEquations import KerasExplainString
        return KerasExplainString(conf)
    if explainer_type == "WhiteBoxExplainer":
        from GEVAI.expost.WhiteBoxExplainer import WhiteBoxExplainer
        return WhiteBoxExplainer(conf)
    elif explainer_type == "Shapely":
        from GEVAI.expost.KerasShapely import KerasShapely
        return KerasShapely(conf)
    elif explainer_type == "LIME":
        from GEVAI.expost.KerasLime import KerasLime
        return KerasLime(conf)
    elif explainer_type == "Metrics":
        from GEVAI.expost.Metrics import Metrics
        return Metrics(conf)
    return None
