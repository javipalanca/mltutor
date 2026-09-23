"""The packaged application must never start loky processes for importance."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from mltutor.viz.features import compute_permutation_importance


def test_permutation_importance_without_child_processes(monkeypatch):
    from joblib.executor import MemmappingExecutor

    def forbidden(*args, **kwargs):
        raise AssertionError("Permutation Importance must not start child processes")

    monkeypatch.setattr(MemmappingExecutor, "get_memmapping_executor", forbidden)
    X, y = load_iris(return_X_y=True)
    model = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    actual = compute_permutation_importance(model, X, y)
    expected = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=1
    )
    np.testing.assert_allclose(actual.importances, expected.importances)
    assert actual.importances.shape == (4, 10)
    assert actual.importances_mean.max() > 0
