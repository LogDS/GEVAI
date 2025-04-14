import math
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt


from GEVAI.expost.ExPost import ExPost


class KerasLime(ExPost):
    def __init__(self, conf):
        self.maxdisplay = conf.maxdisplay
        self.howMuchSample = conf.howMuchSample

    def acceptingType(self):
        from keras.models import Sequential
        return Sequential

    def __call__(self, *args, **kwargs):
        """
        Given a keras sequential model as an input, this function
        uses LIME to explain the predictions on the training data.
        """
        if 'training_x' in kwargs:
            model = args[0]
            training_x = kwargs['training_x']
            training_y = kwargs.get('training_y', None)
            feature_names = kwargs.get('feature_names', [f'feature_{i}' for i in range(training_x.shape[1])])
            class_names = kwargs.get('class_names', None)

            howSample = min(math.ceil(self.howMuchSample * len(training_x)), len(training_x))
            sample_x = training_x[:howSample]

            from GEVAI.utils import fullname
            h = fullname(model)
            mode, predict_fn = None, None
            if h == 'keras.src.models.sequential.Sequential' or h == 'sklearn.tree._classes.DecisionTreeClassifier' or h == 'wittgenstein.ripper.RIPPER':
                try:
                    mode = "classification" if (len(model.output_shape) > 1 and model.output_shape[-1] > 1) else "regression"
                    predict_fn = lambda x: model.predict(x) if mode == "classification" else lambda x: model.predict(x).flatten()
                except Exception as e:
                    print(f"Error during LIME explanation: {e}")

            if mode is not None and predict_fn is not None:
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=sample_x,
                    feature_names=feature_names,
                    class_names=class_names,
                    mode=mode
                )

                num_explanations = min(self.maxdisplay, len(sample_x))
                print(f"Generating LIME explanations for {num_explanations} samples...")

                for i in range(num_explanations):
                    explanation = explainer.explain_instance(
                        data_row=sample_x[i],
                        predict_fn=predict_fn,
                        num_features=self.maxdisplay,
                    )
                    print(f"\n--- Explanation for instance {i} ---")
                    plt.figure()
                    explanation.as_pyplot_figure()
                    plt.tight_layout()
                    plt.savefig(f"lime_explanation_{model.name}_{i}.png")
                    plt.show()
            else:
                print("Unsupported LIME explainer")
