

#random_state=2, verbosity=0, k=2, prune_size=0.33, dl_allowance=64, n_discretize_bins=10

from sklearn.tree import DecisionTreeClassifier
from GEVAI.adhoc.generic_algorithm import GenericAlgorithm


class RipperK_(GenericAlgorithm):
    def __init__(self, conf):
        self.conf = conf

    def __call__(self, *args, **kwargs):
        training_x, training_y = args[0], args[1]
        # training_x = list(df.keys())
        # training_y = [df(x) for x in training_x]
        import wittgenstein as lw
        from sklearn.model_selection import GridSearchCV
        param_grid = {    'random_state': self.conf.RK_Random,
    'k': self.conf.RK_K,
    'prune_size': self.conf.RK_PRUNE,
                          'dl_allowance': self.conf.RK_DL_ALLOWANCE,
                          'n_discretize_bins': self.conf.RK_N_DISCRETIZE_BINS
                          }
        grid_search = GridSearchCV(estimator=lw.RIPPER(),
                                   param_grid=param_grid,
                                   cv=self.conf.FOLD_CROSS_VALIDATION,  # 5-fold cross-validation
                                   scoring=self.conf.METRICS[0])
        grid_search.fit(training_x, training_y)
        print("Best Hyperparameters:", grid_search.best_params_)
        print("Best Accuracy:", grid_search.best_score_)
        r = DecisionTreeClassifier(**grid_search.best_params_)
        r.fit(training_x, training_y)
        return [r]