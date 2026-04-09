"""MI-based feature selection for embedding features."""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)


class MIFeatureSelector:
    """Select features by mutual information with target returns.

    Drops features with MI < threshold, keeping protected columns always.
    Fit on train data, apply same mask to test data.
    """

    def __init__(
        self,
        mi_threshold: float = 0.005,
        protected_columns: Optional[list[str]] = None,
        random_state: int = 42,
    ):
        self.mi_threshold = mi_threshold
        self.protected_columns = protected_columns or []
        self.random_state = random_state
        self.selected_features_: list[str] = []
        self.mi_scores_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MIFeatureSelector":
        """Compute MI scores and select features above threshold."""
        features = np.nan_to_num(X.values, nan=0.0)
        mi = mutual_info_regression(
            features, y, random_state=self.random_state, n_neighbors=5
        )
        self.mi_scores_ = dict(zip(X.columns, mi))

        selected = []
        dropped = []
        for col in X.columns:
            if col in self.protected_columns:
                selected.append(col)
            elif self.mi_scores_[col] >= self.mi_threshold:
                selected.append(col)
            else:
                dropped.append(col)

        self.selected_features_ = selected
        logger.info(
            f"MI feature selection: kept {len(selected)}/{len(X.columns)} features "
            f"(dropped {len(dropped)}: {dropped[:10]}{'...' if len(dropped) > 10 else ''})"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection mask."""
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(X, y)
        return self.transform(X)
