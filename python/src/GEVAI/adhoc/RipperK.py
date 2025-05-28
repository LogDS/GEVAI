# random_state=2, verbosity=0, k=2, prune_size=0.33, dl_allowance=64, n_discretize_bins=10
import pandas as pd

from GEVAI.adhoc import load_model, save_model
from GEVAI.adhoc.RipperKWrapper import RipperKWrapper
from GEVAI.adhoc.generic_algorithm import GenericAlgorithm


class RipperK_(GenericAlgorithm):
    def __init__(self, conf, should_load):
        self.conf = conf
        self.model_filename = "Ripper"
        self.should_load = should_load
        self.target_classes = conf.TARGET_CLASSES
        self.top_n = conf.TOP_N

    def __call__(self, *args, **kwargs):
        training_x, training_y = args[0], args[1]
        # training_x = list(df.keys())
        # training_y = [df(x) for x in training_x]

        trained_model = load_model(self.model_filename, self.should_load)
        if trained_model is None:
            import wittgenstein as lw
            from sklearn.model_selection import GridSearchCV
            from GEVAI.utils import fullname
            w = fullname(lw.RIPPER())

            all_class_models = []

            for target_class in range(self.target_classes):
                print(f'Target class: {target_class}')
                param_grid = {
                    'random_state': self.conf.RK_Random,
                    'k': self.conf.RK_K,
                    'prune_size': self.conf.RK_PRUNE,
                    'dl_allowance': self.conf.RK_DL_ALLOWANCE,
                    'n_discretize_bins': self.conf.RK_N_DISCRETIZE_BINS
                }

                grid_search = GridSearchCV(
                    estimator=lw.RIPPER(),
                    param_grid=param_grid,
                    cv=self.conf.FOLD_CROSS_VALIDATION,  # 5-fold cross-validation
                    scoring=self.conf.METRICS[0]
                )
                grid_search.fit(training_x, training_y == target_class)
                print("Best Hyperparameters:", grid_search.best_params_)
                print("Best Accuracy:", grid_search.best_score_)

                results = pd.DataFrame(grid_search.cv_results_)
                if 'rank_test_score' in results:
                    sorted_results = results.sort_values(by='rank_test_score')
                elif 'mean_test_score' in results:
                    sorted_results = results.sort_values(by='mean_test_score', ascending=False)

                top_models_data = sorted_results.iloc[:self.top_n]
                top_models = []

                for i in range(self.top_n):
                    model = lw.RIPPER(**top_models_data.iloc[i]['params'])
                    model.fit(training_x, training_y == target_class)  # TODO: Is this correct to change training_y to Boolean?
                    top_models.append(model)

                all_class_models.append(top_models)

            num_rows = len(all_class_models)
            num_cols = len(all_class_models[0])
            models = []
            for j in range(num_cols):
                column_values = []
                for i in range(num_rows):
                    column_values.append(all_class_models[i][j])
                models.append(RipperKWrapper(column_values))

            save_model(models, self.model_filename)

            return models
        else:
            if not isinstance(trained_model, RipperKWrapper):
                return [RipperKWrapper(trained_model)]
            return trained_model