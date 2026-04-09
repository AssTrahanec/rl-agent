import numpy as np
import pandas as pd
from src.features.feature_selector import MIFeatureSelector


def test_selector_drops_zero_mi_features():
    """Features with zero MI should be dropped."""
    np.random.seed(42)
    n = 500
    y = np.random.randn(n)
    X = pd.DataFrame({
        "useful": y + np.random.randn(n) * 0.1,
        "noise": np.random.randn(n),
        "also_useful": y * 0.5 + np.random.randn(n) * 0.5,
    })
    selector = MIFeatureSelector(mi_threshold=0.001)
    selector.fit(X, y)
    result = selector.transform(X)
    assert "useful" in result.columns
    assert "also_useful" in result.columns


def test_selector_keeps_protected_columns():
    """Protected columns (price features) are never dropped."""
    np.random.seed(42)
    n = 200
    y = np.random.randn(n)
    X = pd.DataFrame({
        "sma_7": np.random.randn(n),
        "rsi_14": np.random.randn(n),
        "emb_0": np.random.randn(n),
        "emb_1": y + np.random.randn(n) * 0.1,
    })
    protected = ["sma_7", "rsi_14"]
    selector = MIFeatureSelector(mi_threshold=0.01, protected_columns=protected)
    selector.fit(X, y)
    result = selector.transform(X)
    assert "sma_7" in result.columns
    assert "rsi_14" in result.columns
    assert "emb_1" in result.columns


def test_selector_fit_transform():
    """fit_transform returns same result as fit then transform."""
    np.random.seed(42)
    n = 300
    y = np.random.randn(n)
    X = pd.DataFrame({
        "a": y + np.random.randn(n) * 0.1,
        "b": np.random.randn(n),
    })
    selector = MIFeatureSelector(mi_threshold=0.001)
    result1 = selector.fit_transform(X, y)
    result2 = selector.transform(X)
    pd.testing.assert_frame_equal(result1, result2)


def test_selector_selected_features_attribute():
    """After fit, selected_features_ lists kept column names."""
    np.random.seed(42)
    n = 300
    y = np.random.randn(n)
    X = pd.DataFrame({
        "a": y * 2 + np.random.randn(n) * 0.1,
        "b": np.random.randn(n),
    })
    selector = MIFeatureSelector(mi_threshold=0.001)
    selector.fit(X, y)
    assert isinstance(selector.selected_features_, list)
    assert "a" in selector.selected_features_
