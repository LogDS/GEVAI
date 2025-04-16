import math
import shap

from GEVAI.expost.ExPost import ExPost


class KerasShapely(ExPost):

    def __init__(self, conf):
        self.maxdisplay = conf.maxdisplay
        self.howMuchSample = conf.howMuchSample

    def acceptingType(self):
        from keras.models import Sequential
        return Sequential

    def __call__(self, *args, **kwargs):
        """
        Given a keras sequential model as an input, this function
        expresses the neural network as a list of equations
        """
        if 'training_x' in kwargs:
            model = args[0]
            from GEVAI.utils import fullname
            h = fullname(model)
            import matplotlib.pyplot as plt
            training_x = kwargs['training_x']
            howSample = min(math.ceil(self.howMuchSample * len(training_x)), len(training_x))
            if h == 'keras.src.models.sequential.Sequential':
                # shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
                explainer = shap.KernelExplainer(model, training_x[:howSample])
            elif h == 'sklearn.tree._classes.DecisionTreeClassifier' or h == 'wittgenstein.ripper.RIPPER':
                explainer = shap.KernelExplainer(model.predict_proba, training_x[:howSample])
            else:
                print("Unsupported SHAP explainer")
                return False
            shap_values = explainer.shap_values(training_x)
            shap.summary_plot(shap_values, training_x, max_display=self.maxdisplay, show=False)  # .png,.pdf will also support here
            plt.savefig(f"{kwargs['results_path']}/shap_summary_{h}.svg", dpi=700)
            plt.show()
            return True
        print("Unsupported SHAP explainer")
        return False

