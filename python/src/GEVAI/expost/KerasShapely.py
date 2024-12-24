import math

import shap

from GEVAI.expost.ExPost import ExPost


class KerasShapely(ExPost):

    def __init__(self, conf):
        self.maxdisplay = conf.maxdisplay
        self.howMuchSample = conf.howMuchSample

    def acceptingType(self):
        import keras.engine.sequential
        return keras.engine.sequential.Sequential

    def __call__(self, *args, **kwargs):
        """
        Given a keras sequential model as an input, this function
        expresses the neural network as a list of equations
        """
        if 'training_x' in kwargs:
            model = args[0]
            import matplotlib.pyplot as pl
            training_x = kwargs['training_x']
            howSample = min(math.ceil(self.howMuchSample * len(training_x)), len(training_x))
            from GEVAI.utils import fullname
            h = fullname(model)
            if h == 'keras.engine.sequential.Sequential':
                # shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
                explainer = shap.KernelExplainer(model, training_x[:howSample])
                shap_values = explainer.shap_values(training_x)
                shap.summary_plot(shap_values, training_x, max_display=self.maxdisplay,show=False)  # .png,.pdf will also support here
                pl.savefig("shap_summary.svg",dpi=700)
                pl.show()