from GEVAI import (get_ad_hoc_explainer,
                   get_a_priori_explainer,
                   get_ex_post_explainer)

if __name__ == '__main__':
    conf = "/home/giacomo/PyCharmProjects/MLPNAS2/black_box_parameters.yaml"
    ## Loading the configuration for the entire architecture
    conf = get_a_priori_explainer("Configuration", conf)
    ## Loading the dataset
    ## TODO: use the actual representation hierarchy
    df = get_a_priori_explainer("PandasLoad", conf)
    ## Use "MLPNAS_pretrained" if you already run the model over the dataset, so to use the persisted outcome
    ## of the model being trained, ad MLPNAS if you need to train this for the first time
    ad_hoc = get_ad_hoc_explainer("MLPNAS_pretrained", conf)
    for model in ad_hoc(*df):
        ## Obtaining the equations explaining the neural network as mathematical functions
        ex_post1 = get_ex_post_explainer("BlackBoxExplainer", conf)
        explanation1 = ex_post1(model, training_x=df[0])

        ## Obtaining the shapely value explanation for the model
        ex_post2 = get_ex_post_explainer("ModelAgnostic", conf)
        explanation2 = ex_post2(model, training_x=df[0])
