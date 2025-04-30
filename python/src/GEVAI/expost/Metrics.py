import json
import math

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from GEVAI.benchmarking import write_to_file
from GEVAI.expost.ExPost import ExPost


class Metrics(ExPost):
    def __init__(self, conf):
        self.maxdisplay = conf.maxdisplay
        self.howMuchSample = conf.howMuchSample

    def __call__(self, *args, **kwargs):
        if 'training_x' in kwargs and 'training_y' in kwargs:
            model = args[0]
            training_x = kwargs['training_x']
            training_y = kwargs['training_y']
            howSample = min(math.ceil(self.howMuchSample * len(training_x)), len(training_x))

            from GEVAI.utils import fullname
            h = fullname(model)

            train_x, test_x, train_y, test_y = train_test_split(training_x, training_y, test_size=0.2, random_state=42)

            # if hasattr(model, "predict_proba"):
            #     predictions = model.predict_proba(test_x)
            # else:
            predictions = model.predict(test_x)

            # For binary classification with sigmoid activation (output probabilities between 0 and 1):
            if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                predicted_classes = (predictions > 0.5).astype(int).flatten()
            # For multi-class classification with softmax activation (output probabilities for each class):
            elif len(predictions.shape) > 1 and predictions.shape[1] > 1:
                predicted_classes = np.argmax(predictions, axis=1)
            # For binary classification where the model directly outputs 0 or 1:
            else:
                predicted_classes = predictions.astype(int).flatten()

            # a 1D array of class labels
            true_labels = test_y.astype(int).flatten()

            # For binary classification:
            if len(np.unique(true_labels)) == 2:
                accuracy = accuracy_score(true_labels, predicted_classes)
                precision = precision_score(true_labels, predicted_classes)
                recall = recall_score(true_labels, predicted_classes)
                f1 = f1_score(true_labels, predicted_classes)

                print("--- Binary Classification Metrics ---")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")

                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                }
                write_to_file(f"{kwargs['results_path']}/metrics_{h}.json", json.dumps(metrics), 'w')
                return metrics
            # For multi-class classification:
            else:
                print("--- Multi-class Classification Metrics ---")

                # Macro-average (average of per-class metrics)
                precision_macro = precision_score(true_labels, predicted_classes, average='macro')
                recall_macro = recall_score(true_labels, predicted_classes, average='macro')
                f1_macro = f1_score(true_labels, predicted_classes, average='macro')
                print(f"Macro-average Precision: {precision_macro:.4f}")
                print(f"Macro-average Recall: {recall_macro:.4f}")
                print(f"Macro-average F1 Score: {f1_macro:.4f}")

                # Weighted-average (average of per-class metrics weighted by support)
                precision_weighted = precision_score(true_labels, predicted_classes, average='weighted')
                recall_weighted = recall_score(true_labels, predicted_classes, average='weighted')
                f1_weighted = f1_score(true_labels, predicted_classes, average='weighted')
                print(f"Weighted-average Precision: {precision_weighted:.4f}")
                print(f"Weighted-average Recall: {recall_weighted:.4f}")
                print(f"Weighted-average F1 Score: {f1_weighted:.4f}")

                accuracy = accuracy_score(true_labels, predicted_classes)
                print(f"Accuracy: {accuracy:.4f}")

                metrics = {
                    'accuracy': accuracy,
                    'precision': {
                        'macro': precision_macro,
                        'weighted': precision_weighted
                    },
                    'recall': {
                        'macro': recall_macro,
                        'weighted': recall_weighted
                    },
                    'f1_score': {
                        'macro': f1_macro,
                        'weighted': f1_weighted
                    },
                }
                write_to_file(f"{kwargs['results_path']}/metrics_{h}.json", json.dumps(metrics), 'w')
                return metrics
        else:
            return False
