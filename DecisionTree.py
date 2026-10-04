import numpy as np
import pandas as pd


class Node:
    def __init__(
        self,
        target,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_numeric=False,
    ):
        self.target = target
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_numeric = is_numeric
        self.is_leaf = left_child is None and right_child is None

        if self.is_leaf:
            self.prediction = self._get_majority_class()
        else:
            self.prediction = None

    def _get_majority_class(self):
        unique, counts = np.unique(self.target, return_counts=True)
        winner_index = counts.argmax()
        return unique[winner_index]

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf: prediction={self.prediction}"
        if isinstance(self.threshold, (int, float)):
            return f"Node: '{self.feature}' <= {self.threshold:.2f}"
        return f"Node: '{self.feature}' == {self.threshold}"


class DecisionTree:
    def __init__(self, root=None, max_depth=3):
        self.max_depth = max_depth
        self.root = root

    def __repr__(self):
        if self.root is None:
            return "<empty tree>"
        return self._repr_recursive(self.root, prefix="")

    def _calculate_entropy(self, labels=np.array) -> float:
        """Compute label impurity. 0 = perfectly pure, 1 = maximum entropy."""
        unique_labels, counts = np.unique(labels, return_counts=True)

        if len(unique_labels) <= 1:
            return 0.0

        entropy = 0
        total = len(labels)

        for c in counts:
            chance = c / total
            entropy += chance * np.log2(chance)

        return -entropy

    def _calculate_information_gain(self, parent_labels: np.array, child_labels: list) -> float:
        """
        Measure improvement from a split.

        Information Gain = (parent entropy) - (weighted average child entropy).
        """
        entropy_parent = self._calculate_entropy(parent_labels)
        entropy_left_child = self._calculate_entropy(child_labels[0])
        entropy_right_child = self._calculate_entropy(child_labels[1])

        n_total = len(parent_labels)

        weight_left_child = len(child_labels[0]) / n_total
        weight_right_child = len(child_labels[1]) / n_total
        weighted_mean = (
            weight_left_child * entropy_left_child
            + weight_right_child * entropy_right_child
        )

        return entropy_parent - weighted_mean

    def _split_data(
        self,
        features: pd.DataFrame,
        target: np.array,
        feature: str,
        threshold: any,
        is_numeric: bool = False,
    ) -> list:
        """
        Split data based on a feature and threshold.

        Args:
            features: DataFrame with features.
            target: Series with target labels.
            feature: Feature to split on.
            threshold: Threshold value for the split.
            is_numeric: Whether the feature is numeric or categorical.

        Returns:
            List with target and feature subsets for left and right children.
        """
        if is_numeric:
            mask_left = features[feature] <= threshold
            mask_right = features[feature] > threshold
        else:
            mask_left = features[feature] == threshold
            mask_right = features[feature] != threshold

        target_left = target[mask_left]
        target_right = target[mask_right]
        features_left = features[mask_left]
        features_right = features[mask_right]

        return [target_left, target_right, features_left, features_right]

    def _find_best_split(self, features: pd.DataFrame, target: pd.Series) -> tuple:
        """
        Find the best split for the given features and target.

        Returns:
            Tuple of feature name and threshold value for the best split.
        """
        max_gain = 0
        best_feature = None
        best_threshold = None

        for feature in features.columns:
            unique_values = np.unique(features[feature])
            unique_values.sort()

            is_numeric_col = pd.api.types.is_numeric_dtype(features[feature])

            if is_numeric_col:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            else:
                thresholds = unique_values

            for threshold in thresholds:
                split = self._split_data(
                    features, target, feature, threshold, is_numeric=is_numeric_col
                )
                gain = self._calculate_information_gain(target, split)

                if gain > max_gain:
                    best_feature = feature
                    best_threshold = threshold
                    max_gain = gain

        return best_feature, best_threshold

    def _build_recursive(self, features, target, depth):
        """Recursively build the tree for left and right branches."""
        if len(np.unique(target)) == 1 or depth >= self.max_depth:
            return Node(target)

        best_feature, best_threshold = self._find_best_split(features, target)

        if best_feature is None:
            return Node(target)

        is_num = pd.api.types.is_numeric_dtype(features[best_feature])
        t_l, t_r, f_l, f_r = self._split_data(
            features, target, best_feature, best_threshold, is_numeric=is_num
        )

        left_child = self._build_recursive(f_l, t_l, depth=depth + 1)
        right_child = self._build_recursive(f_r, t_r, depth=depth + 1)

        return Node(
            target,
            best_feature,
            best_threshold,
            left_child=left_child,
            right_child=right_child,
            is_numeric=is_num,
        )

    def _repr_recursive(self, node: Node, prefix="", type=None) -> str:
        """Return a string representation of the subtree rooted at `node`."""
        if node is None:
            return ""

        if type == 'left':
            line_prefix = prefix + "├── L: "
            new_prefix = prefix + "│   "
        elif type == 'right':
            line_prefix = prefix + "└── R: "
            new_prefix = prefix + "    "
        else:
            line_prefix = ""
            new_prefix = prefix

        lines = [line_prefix + str(node)]

        left_str = self._repr_recursive(node.left_child, new_prefix, type='left')
        right_str = self._repr_recursive(node.right_child, new_prefix, type='right')

        if left_str:
            lines.append(left_str)
        if right_str:
            lines.append(right_str)

        return "\n".join(lines)

    def _predict_recursive(self, node, row):
        """Navigate a single row through the tree until a leaf is reached."""
        if node.is_leaf:
            return node.prediction

        val = row[node.feature]

        if node.is_numeric:
            go_left = val <= node.threshold
        else:
            go_left = val == node.threshold

        if go_left:
            return self._predict_recursive(node.left_child, row)
        return self._predict_recursive(node.right_child, row)

    def fit(self, features, target):
        self.root = self._build_recursive(features, target, depth=0)

    def predict(self, df):
        predictions = []
        for _, row in df.iterrows():
            result = self._predict_recursive(self.root, row)
            predictions.append(result)
        return np.array(predictions)
