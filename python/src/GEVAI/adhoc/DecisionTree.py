from sklearn.tree import DecisionTreeClassifier
from GEVAI.adhoc.generic_algorithm import GenericAlgorithm


class DecisionTree_(GenericAlgorithm):
    def __init__(self, conf):
        self.conf = conf

    def __call__(self, *args, **kwargs):
        training_x, training_y = args[0], args[1]
        # training_x = list(df.keys())
        # training_y = [df(x) for x in training_x]
        from sklearn.model_selection import GridSearchCV
        param_grid = {    'max_depth': self.conf.DT_MAX_DEPTH,
    'min_samples_split': self.conf.DT_MIN_SAMPLES_SPLIT,
    'criterion': self.conf.DT_CRITERION
                          }
        grid_search = GridSearchCV(estimator=DecisionTreeClassifier(),
                                   param_grid=param_grid,
                                   cv=self.conf.FOLD_CROSS_VALIDATION,  # 5-fold cross-validation
                                   scoring=self.conf.METRICS[0])
        grid_search.fit(training_x, training_y)
        print("Best Hyperparameters:", grid_search.best_params_)
        print("Best Accuracy:", grid_search.best_score_)
        r = DecisionTreeClassifier(**grid_search.best_params_)
        r.fit(training_x, training_y)
        return [r]