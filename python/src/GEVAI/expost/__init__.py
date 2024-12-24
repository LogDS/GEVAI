
def get_ex_post_explainer(explainer_type, conf):
    if explainer_type == "BlackBoxExplainer":
        from GEVAI.expost.KerasEquations import KerasExplainString
        return KerasExplainString(conf)
    elif explainer_type == "ModelAgnostic":
        from GEVAI.expost.KerasShapely import KerasShapely
        return KerasShapely(conf)
