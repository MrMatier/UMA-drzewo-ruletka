import numpy as np
from collections import Counter

class TreeNode:
    def __init__(self, is_leaf=False, prediction=None, feature=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.children = {}

class DecisionTreeModel:
    def __init__(self, mode="greedy", max_depth=10, min_samples_split=2, random_state=None):
        self.mode = mode
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.root = None
        self.default_class = None
        self.rng = np.random.default_rng(random_state)

        if mode not in ["greedy", "roulette"]:
            raise ValueError("mode must be 'greedy' or 'roulette'")

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.default_class = self._most_common_class(y)
        features = list(range(X.shape[1]))
        self.root = self._build_tree(X, y, features, depth=0)
        return self

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_one(row, self.root) for row in X])

    def _build_tree(self, X, y, features, depth):
        current_prediction = self._most_common_class(y)       #najczęstsza klasa w węźle używana jako predykcja awaryjna

        #warunki stop
        if len(set(y)) == 1:
            return TreeNode(is_leaf=True, prediction=y[0])

        if len(features) == 0:
            return TreeNode(is_leaf=True, prediction=current_prediction)

        if self.max_depth is not None and depth >= self.max_depth:
            return TreeNode(is_leaf=True, prediction=current_prediction)

        if len(y) < self.min_samples_split:
            return TreeNode(is_leaf=True, prediction=current_prediction)

        gains = []
        for feature in features:
            gain = self._info_gain(X, y, feature)
            gains.append(gain)

        gains = np.array(gains)

        #brak sensownego podziału
        if np.all(gains <= 0):
            return TreeNode(is_leaf=True, prediction=current_prediction)

        if self.mode == "greedy":
            chosen_feature = self._choose_greedy(features, gains)
        else:
            chosen_feature = self._choose_roulette(features, gains)

        node = TreeNode(is_leaf=False, prediction=current_prediction, feature=chosen_feature)

        values = np.unique(X[:, chosen_feature])
        new_features = [f for f in features if f != chosen_feature]

        for value in values:
            mask = X[:, chosen_feature] == value
            child_X = X[mask]
            child_y = y[mask]

            if len(child_y) == 0:
                child = TreeNode(is_leaf=True, prediction=current_prediction)
            else:
                child = self._build_tree(child_X, child_y, new_features, depth + 1)

            node.children[value] = child

        return node

    def _predict_one(self, row, node):
        while not node.is_leaf:
            value = row[node.feature]

            #jeśli w testowym pojawi wartość której nie było w treningu
            if value not in node.children:
                return node.prediction

            node = node.children[value]

        return node.prediction

    def _entropy(self, y):
        counts = Counter(y)
        total = len(y)

        result = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                result -= p * np.log2(p)

        return result

    def _info_gain(self, X, y, feature):
        base_entropy = self._entropy(y)
        values = np.unique(X[:, feature])

        weighted_entropy = 0.0

        for value in values:
            mask = X[:, feature] == value
            part_y = y[mask]
            weight = len(part_y) / len(y)
            weighted_entropy += weight * self._entropy(part_y)

        return base_entropy - weighted_entropy

    def _choose_greedy(self, features, gains):
        best_index = np.argmax(gains)
        return features[best_index]

    def _choose_roulette(self, features, gains):
        gains = np.maximum(gains, 0)
        total = np.sum(gains)

        #gdyby wszystkie wartości były 0 losujemy zwyczajnie
        if total == 0:
            return self.rng.choice(features)

        probabilities = gains / total
        return self.rng.choice(features, p=probabilities)

    def _most_common_class(self, y):
        return Counter(y).most_common(1)[0][0]

    def get_depth(self):
        return self._get_depth(self.root)

    def _get_depth(self, node):
        if node is None:
            return 0

        if node.is_leaf:
            return 0

        if len(node.children) == 0:
            return 0

        return 1 + max(self._get_depth(child) for child in node.children.values())

    def count_leaves(self):
        return self._count_leaves(self.root)

    def _count_leaves(self, node):
        if node is None:
            return 0

        if node.is_leaf:
            return 1

        return sum(self._count_leaves(child) for child in node.children.values())