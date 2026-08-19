from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_float_matrix(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


class MedianImputer:
    def fit(self, X: Any) -> "MedianImputer":
        matrix = _as_float_matrix(X)
        self.medians_ = np.nanmedian(matrix, axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        return self

    def transform(self, X: Any) -> np.ndarray:
        matrix = _as_float_matrix(X).copy()
        rows, columns = np.where(np.isnan(matrix))
        matrix[rows, columns] = self.medians_[columns]
        return matrix


class LinearRegressorLite:
    """Régression linéaire standardisée, basée uniquement sur NumPy."""

    def fit(self, X: Any, y: Any) -> "LinearRegressorLite":
        self.imputer_ = MedianImputer().fit(X)
        matrix = self.imputer_.transform(X)
        target = np.asarray(y, dtype=np.float64)
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        standardized = (matrix - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(standardized)), standardized])
        self.coefficients_, *_ = np.linalg.lstsq(design, target, rcond=None)
        return self

    def predict(self, X: Any) -> np.ndarray:
        matrix = self.imputer_.transform(X)
        standardized = (matrix - self.mean_) / self.scale_
        return self.coefficients_[0] + standardized @ self.coefficients_[1:]


@dataclass
class TreeNode:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class RegressionTreeLite:
    def __init__(
        self,
        *,
        max_depth: int = 8,
        min_samples_leaf: int = 12,
        max_features: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    @staticmethod
    def _sse(values: np.ndarray) -> float:
        if len(values) == 0:
            return float("inf")
        return float(np.sum((values - values.mean()) ** 2))

    def fit(self, X: Any, y: Any) -> "RegressionTreeLite":
        matrix = _as_float_matrix(X)
        target = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = matrix.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.root_ = self._build(matrix, target, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        node = TreeNode(value=float(y.mean()))
        if (
            depth >= self.max_depth
            or len(y) < 2 * self.min_samples_leaf
            or float(np.var(y)) < 1e-10
        ):
            return node

        feature_count = self.max_features or max(1, int(np.sqrt(self.n_features_in_)))
        feature_count = min(feature_count, self.n_features_in_)
        features = self._rng.choice(self.n_features_in_, size=feature_count, replace=False)
        best_score = float("inf")
        best: tuple[int, float, np.ndarray] | None = None

        for feature in features:
            values = X[:, feature]
            thresholds = np.unique(np.quantile(values, [0.15, 0.30, 0.50, 0.70, 0.85]))
            for threshold in thresholds:
                left_mask = values <= threshold
                left_count = int(left_mask.sum())
                if left_count < self.min_samples_leaf or len(y) - left_count < self.min_samples_leaf:
                    continue
                score = self._sse(y[left_mask]) + self._sse(y[~left_mask])
                if score < best_score:
                    best_score = score
                    best = (int(feature), float(threshold), left_mask)

        if best is None:
            return node
        feature, threshold, left_mask = best
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def predict(self, X: Any) -> np.ndarray:
        matrix = _as_float_matrix(X)
        output = np.empty(len(matrix), dtype=np.float64)

        def fill(node: TreeNode, indices: np.ndarray) -> None:
            if not len(indices):
                return
            if node.is_leaf:
                output[indices] = node.value
                return
            assert node.feature is not None and node.threshold is not None
            assert node.left is not None and node.right is not None
            left = matrix[indices, node.feature] <= node.threshold
            fill(node.left, indices[left])
            fill(node.right, indices[~left])

        fill(self.root_, np.arange(len(matrix)))
        return output


class RandomForestLiteRegressor:
    """Forêt de CART aléatoires compacte pour la preuve de concept locale."""

    def __init__(
        self,
        *,
        n_estimators: int = 36,
        max_depth: int = 9,
        min_samples_leaf: int = 12,
        max_samples: int = 8_000,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "RandomForestLiteRegressor":
        self.imputer_ = MedianImputer().fit(X)
        matrix = self.imputer_.transform(X)
        target = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        sample_size = min(self.max_samples, len(matrix))
        self.trees_: list[RegressionTreeLite] = []
        for index in range(self.n_estimators):
            sample = rng.integers(0, len(matrix), size=sample_size)
            tree = RegressionTreeLite(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + index + 1,
            )
            tree.fit(matrix[sample], target[sample])
            self.trees_.append(tree)
        return self

    def predict(self, X: Any) -> np.ndarray:
        matrix = self.imputer_.transform(X)
        predictions = np.vstack([tree.predict(matrix) for tree in self.trees_])
        return predictions.mean(axis=0)


class GradientBoostingLiteRegressor:
    """Boosting séquentiel de petits arbres de régression."""

    def __init__(
        self,
        *,
        n_estimators: int = 55,
        learning_rate: float = 0.06,
        max_depth: int = 3,
        min_samples_leaf: int = 18,
        max_samples: int = 10_000,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "GradientBoostingLiteRegressor":
        self.imputer_ = MedianImputer().fit(X)
        matrix = self.imputer_.transform(X)
        target = np.asarray(y, dtype=np.float64)
        self.initial_ = float(target.mean())
        prediction = np.full(len(target), self.initial_, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        self.trees_: list[RegressionTreeLite] = []
        sample_size = min(self.max_samples, len(matrix))
        for index in range(self.n_estimators):
            residual = target - prediction
            sample = rng.choice(len(matrix), size=sample_size, replace=False)
            tree = RegressionTreeLite(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=matrix.shape[1],
                random_state=self.random_state + 10_000 + index,
            )
            tree.fit(matrix[sample], residual[sample])
            prediction += self.learning_rate * tree.predict(matrix)
            self.trees_.append(tree)
        return self

    def predict(self, X: Any) -> np.ndarray:
        matrix = self.imputer_.transform(X)
        prediction = np.full(len(matrix), self.initial_, dtype=np.float64)
        for tree in self.trees_:
            prediction += self.learning_rate * tree.predict(matrix)
        return prediction
