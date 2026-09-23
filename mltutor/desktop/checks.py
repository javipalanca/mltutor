"""Small real-model checks runnable inside the portable executable.

No pytest, development environment, network or user datasets are needed.
"""

from pathlib import Path
import pickle
import tempfile
import numpy as np


def verify_models():
    from sklearn.datasets import load_iris, load_diabetes
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsClassifier
    from mltutor.algorithms.model_training import train_neural_network
    from mltutor.utils import export_model_onnx
    from mltutor.algorithms.export import (
        export_saved_model_as_zip,
        convert_keras_to_tflite,
    )
    import onnx
    import tensorflow as tf
    import pandas as pd

    iris = load_iris()
    X, y = iris.data, iris.target
    tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
    assert tree.predict(X[:2]).shape == (2,)
    restored = pickle.loads(pickle.dumps(tree))
    np.testing.assert_array_equal(tree.predict(X), restored.predict(X))
    encoded = export_model_onnx(tree, X.shape[1])
    assert encoded, "La exportación ONNX no produjo datos"
    onnx.checker.check_model(onnx.load_from_string(encoded))
    knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    assert knn.predict(X[:2]).shape == (2,)
    diabetes = load_diabetes()
    regression = LinearRegression().fit(diabetes.data, diabetes.target)
    assert np.isfinite(regression.predict(diabetes.data[:2])).all()
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["target"] = y
    config = dict(
        input_size=4,
        output_size=3,
        architecture=[4, 4, 3],
        activation="relu",
        output_activation="softmax",
        dropout_rate=0.0,
        optimizer="adam",
        batch_size=16,
        task_type="Clasificación",
    )
    model, history, X_test, _, _, _ = train_neural_network(
        df, "target", config, 0.01, 2, 0.2, False, None, False, None
    )
    assert model is not None, "El entrenamiento neuronal falló"
    prediction = model.predict(X_test[:2], verbose=0)
    assert prediction.shape == (2, 3)
    assert np.isfinite(prediction).all()
    with tempfile.TemporaryDirectory(prefix="mltutor-check-") as folder:
        path = Path(folder) / "network.keras"
        model.save(path)
        loaded = tf.keras.models.load_model(path)
        np.testing.assert_allclose(
            loaded.predict(X_test[:2], verbose=0), prediction, rtol=1e-5
        )
    archive = export_saved_model_as_zip(model, True)
    assert archive.startswith(b"PK"), "SavedModel no produjo un ZIP"
    lite_data = convert_keras_to_tflite(model)
    assert lite_data[4:8] == b"TFL3", "TFLite no produjo un modelo válido"
    interpreter = tf.lite.Interpreter(model_content=lite_data)
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    interpreter.set_tensor(inputs[0]["index"], X_test[:1].astype(np.float32))
    interpreter.invoke()
    np.testing.assert_allclose(
        interpreter.get_tensor(outputs[0]["index"]),
        prediction[:1],
        rtol=1e-4,
        atol=1e-5,
    )
    print(
        "MLTutor: árboles, regresión, KNN, red neuronal, Pickle, ONNX, SavedModel y TFLite correctos."
    )
