import numpy as np

class RipperKWrapper:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, data):
        all_predictions = []
        for model in self.models:
            all_predictions.append(model.predict_proba(data)[:,1])

        predictions = np.array(all_predictions).transpose()
        return predictions

    def predict(self, data):
        predictions = self.predict_proba(data)
        return np.argmax(predictions, axis=1)