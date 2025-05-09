from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import _tree

from GEVAI.benchmarking import write_to_file
from GEVAI.expost.ExPost import ExPost


def export_text2(decision_tree, feature_names=None,
                spacing=3, decimals=5, show_weights=False):
    """Build a text report showing the rules of a decision tree.

    Note that backwards compatibility may not be supported.

    Parameters
    ----------
    decision_tree : object
        The decision tree estimator to be exported.
        It can be an instance of
        DecisionTreeClassifier or DecisionTreeRegressor.

    feature_names : list of str, default=None
        A list of length n_features containing the feature names.
        If None generic names will be used ("feature_0", "feature_1", ...).

    max_depth : int, default=10
        Only the first max_depth levels of the tree are exported.
        Truncated branches will be marked with "...".

    spacing : int, default=3
        Number of spaces between edges. The higher it is, the wider the result.

    decimals : int, default=2
        Number of decimal digits to display.

    show_weights : bool, default=False
        If true the classification weights will be exported on each leaf.
        The classification weights are the number of samples each class.

    Returns
    -------
    report : string
        Text summary of all the rules in the decision tree.

    Examples
    --------

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.tree import export_text
    >>> iris = load_iris()
    >>> X = iris['data']
    >>> y = iris['target']
    >>> decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
    >>> decision_tree = decision_tree.fit(X, y)
    >>> r = export_text(decision_tree, feature_names=iris['feature_names'])
    >>> print(r)
    |--- petal width (cm) <= 0.80
    |   |--- class: 0
    |--- petal width (cm) >  0.80
    |   |--- petal width (cm) <= 1.75
    |   |   |--- class: 1
    |   |--- petal width (cm) >  1.75
    |   |   |--- class: 2
    """
    tree_ = decision_tree.tree_
    class_names = decision_tree.classes_
    right_child_fmt = "{} {} <= {}\n"
    left_child_fmt = "{} {} >  {}\n"
    truncation_fmt = "{} {}\n"

    if (feature_names is not None and
            len(feature_names) != tree_.n_features):
        raise ValueError("feature_names must contain "
                         "%d elements, got %d" % (tree_.n_features,
                                                  len(feature_names)))

    if spacing <= 0:
        raise ValueError("spacing must be > 0, given %d" % spacing)

    if decimals < 0:
        raise ValueError("decimals must be >= 0, given %d" % decimals)

    from sklearn.tree import DecisionTreeClassifier
    if isinstance(decision_tree, DecisionTreeClassifier):
        value_fmt = "{}{} weights: {}\n"
        if not show_weights:
            value_fmt = "{}{}{}\n"
    else:
        value_fmt = "{}{} value: {}\n"

    if feature_names is not None:
        feature_names_ = [feature_names[i] if i != _tree.TREE_UNDEFINED
                          else None for i in tree_.feature]
    else:
        feature_names_ = ["feature_{}".format(i) for i in tree_.feature]

    #export_text.report = ""

    def print_tree_recurse(node, depth, acc):
        #indent = ("|" + (" " * spacing)) * depth
        #indent = indent[:-spacing] + "-" * spacing

        if True:
            #info_fmt = ""
            #info_fmt_left = info_fmt
            #info_fmt_right = info_fmt

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                rules = []
                name = feature_names_[node]
                threshold = tree_.threshold[node]
                threshold = "{1:.{0}f}".format(decimals, threshold)
                rhs = "(" + right_child_fmt.format("", name, threshold).replace('\n', '').replace('\r', '').strip() + ")"
                #export_text.report += info_fmt_left
                if (len(acc) > 0):
                    rhs = " and " + rhs
                rules.extend(print_tree_recurse(tree_.children_left[node], depth+1, acc + rhs))

                lhs =  "(" + left_child_fmt.format("", name, threshold).replace('\n', '').replace('\r', '').strip() +")"
                if (len(acc) > 0):
                    lhs = " and " + lhs
                #export_text.report += info_fmt_right
                rules.extend(print_tree_recurse(tree_.children_right[node], depth+1, acc + lhs))
                return rules
            else:  # leaf
                value = None
                if tree_.n_outputs == 1:
                   value = tree_.value[node][0]
                else:
                   value = tree_.value[node].T[0]
                import numpy as np
                class_name = np.argmax(value)

                if (tree_.n_classes[0] != 1 and tree_.n_outputs == 1):
                   class_name = class_names[class_name]
                return [acc + " => Label=" + str(class_name) + " (0/0)"]
        # else:
        #     subtree_depth = _compute_depth(tree_, node)
        #     if subtree_depth == 1:
        #         _add_leaf(value, class_name, indent)
        #     else:
        #         trunc_report = 'truncated branch of depth %d' % subtree_depth
        #         export_text.report += truncation_fmt.format(indent,
        #                                                     trunc_report)

    return print_tree_recurse(0, 1, "")


class WhiteBoxExplainer(ExPost):

    def __init__(self, conf):
        self.conf = conf

    def acceptingType(self):
        pass

    def __call__(self, *args, **kwargs):
        """
        Given a keras sequential model as an input, this function
        expresses the neural network as a list of equations
        """
        training_x = kwargs['training_x']
        training_y = kwargs.get('training_y', None)
        feature_names = kwargs.get('feature_names', [f'feature_{i}' for i in range(training_x.shape[1])])
        class_names = kwargs.get('class_names', None)
        hypothesis = args[0]

        ## TODO: do something depending on the type!
        input_values = None
        from GEVAI.utils import fullname
        h = fullname(hypothesis)
        file_path = f"{kwargs['results_path']}/WhiteBoxExplainer_{h}.txt"
        if h == 'sklearn.tree._classes.DecisionTreeClassifier':
            fig = plt.figure(figsize=(25, 20))
            _ = tree.plot_tree(hypothesis,
                               feature_names=feature_names,
                               class_names=class_names,
                               filled=True)
            fig.savefig(f"{kwargs['results_path']}/DecisionTree_{h}.png")
            write_to_file(file_path, [export_text2(hypothesis)], 'w')
            return [export_text2(hypothesis)]
        elif h == 'wittgenstein.ripper.RIPPER' or h == 'GEVAI.adhoc.RipperKWrapper.RipperKWrapper':
            ruleset_str = ""
            models = hypothesis.models if hasattr(hypothesis, "models") else [hypothesis]
            for m in models:
                ruleset_str += (
                    str([str(rule) for rule in m.ruleset_.rules])
                    .replace(" ", "")
                    .replace(",", " V\n")
                    .replace("'", "")
                    .replace("^", " ^ ")
                )
            write_to_file(file_path, [ruleset_str], 'w')
            return [ruleset_str]
        else:
            print("Unsupported WhiteBox explainer")
            return []