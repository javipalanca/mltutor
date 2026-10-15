"""Executable specification of what the 🧠 Redes Neuronales lesson computes (fase5-redes.md §3, C21–C25).

Pure NumPy, no TensorFlow. ``tools/golden_nn.py`` checks every function here against the
reference app (scikit-learn 1.7.2, Keras 3.12 on TensorFlow 2.16 CPU) and stores the result
of that check in the fixtures, so ``crates/mltutor-nn`` can follow this file line by line:

* ``stratified_split``: ``train_test_split(test_size=0.2, random_state=42, stratify=y)``
  (``StratifiedShuffleSplit._iter_indices`` + ``_approximate_mode``) using only
  ``RandomState.permutation``, which Rust already reproduces with ``rand_mt`` (MEMORY.md).
* ``standard_scaler``: ``StandardScaler().fit_transform(X)`` on ALL the rows (the app scales
  before splitting).
* ``keras_validation_split``: ``model.fit(validation_split=v)`` keeps the first
  ``floor(n·(1 − v))`` rows for training and the LAST rows for validation (no shuffling).
* ``train``: the full ``model.fit`` loop with ``shuffle=False`` (level (a) of the parity
  strategy): dense layers, activations, the loss Keras resolves for each output activation
  (cross-entropies from the logits, as Keras does when the output is sigmoid/softmax), the
  accuracy Keras resolves (binary/categorical), Adam/SGD/RMSprop with Keras 3 formulas, epoch
  logs as batch-size-weighted means, validation at the end of every epoch, EarlyStopping
  (restores the best weights at the end of training, even without stopping early) and
  ReduceLROnPlateau (min_delta 1e-4, patience max(1, patience // 2), min_lr 1e-7).

Everything runs in float32 like Keras; matrix products accumulate in float32 too, so the
results differ from TensorFlow only by summation order (see the tolerances in §6.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

F32 = np.float32
EPS_KERAS = 1e-7  # keras.backend.epsilon()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def _approximate_mode(class_counts: np.ndarray, n_draws: int, rng: np.random.RandomState) -> np.ndarray:
    """sklearn.utils._approximate_mode (legacy RandomState.choice = permutation(n)[:size])."""
    continuous = class_counts / class_counts.sum() * n_draws
    floored = np.floor(continuous)
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = np.sort(np.unique(remainder))[::-1]
        for value in values:
            (inds,) = np.where(remainder == value)
            add_now = min(len(inds), need_to_add)
            inds = inds[rng.permutation(len(inds))[:add_now]]
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(int)


def stratified_split(y, test_size: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of ``train_test_split(..., stratify=y)`` (train, test), in sklearn's order."""
    y = np.asarray(y)
    n = len(y)
    n_test = math.ceil(test_size * n)
    n_train = n - n_test
    classes, y_indices = np.unique(y, return_inverse=True)
    class_counts = np.bincount(y_indices)
    class_indices = np.split(np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1])
    rng = np.random.RandomState(seed)
    n_i = _approximate_mode(class_counts, n_train, rng)
    t_i = _approximate_mode(class_counts - n_i, n_test, rng)
    train: list[int] = []
    test: list[int] = []
    for i in range(len(classes)):
        perm = rng.permutation(class_counts[i])
        idx = class_indices[i].take(perm, mode="clip")
        train.extend(idx[: n_i[i]])
        test.extend(idx[n_i[i]: n_i[i] + t_i[i]])
    train_arr = np.asarray(train)[rng.permutation(len(train))]
    test_arr = np.asarray(test)[rng.permutation(len(test))]
    return train_arr, test_arr


def plain_split(n: int, test_size: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """``train_test_split`` without stratification (regression): D7, ``split.rs``."""
    n_test = math.ceil(test_size * n)
    perm = np.random.RandomState(seed).permutation(n)
    return perm[n_test:], perm[:n_test]


def standard_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``StandardScaler`` (population std, scale 1 for constant columns): (X_scaled, mean, scale)."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    var = ((X - mean) ** 2).mean(axis=0)
    scale = np.sqrt(var)
    scale[scale == 0.0] = 1.0
    return (X - mean) / scale, mean, scale


def keras_validation_split(n_train: int, validation_split: float) -> int:
    """Rows kept for training; the rest (the last ones) are the validation set."""
    return int(math.floor(n_train * (1.0 - validation_split)))


# ---------------------------------------------------------------------------
# Loss and metric resolution (mltutor/algorithms/model_training.py::train_neural_network)
# ---------------------------------------------------------------------------
def resolve_training(task: str, output_activation: str, output_size: int) -> dict[str, Any]:
    """What the app compiles: target encoding, loss, metric and the error it hits (if any)."""
    if task == "Clasificación":
        one_hot = output_activation == "softmax" or (output_size > 1 and output_activation != "sigmoid")
        if output_activation == "sigmoid":
            loss = "binary_crossentropy"
        elif output_activation == "softmax":
            loss = "sparse_categorical_crossentropy" if output_size == 1 else "categorical_crossentropy"
        elif output_activation == "linear":
            loss = "sparse_categorical_crossentropy"
        else:  # tanh
            loss = "mse"
        y_last = output_size if one_hot else 1
        y_rank = 2 if one_hot else 1
        # Keras: is_binary = y_pred.shape[-1] == 1; sparse = rank(y) < rank(pred) or ...
        if output_size == 1:
            metric = "binary_accuracy"
        elif y_rank < 2:
            metric = "sparse_categorical_accuracy"
        else:
            metric = "categorical_accuracy"
        error = None
        if loss == "binary_crossentropy" and output_size > 1:
            error = "rank"  # y (n,) vs output (n, k): "Arguments `target` and `output` must have the same rank"
        elif loss == "sparse_categorical_crossentropy" and one_hot:
            error = "sparse_rank"  # one-hot y with a sparse loss
        elif loss == "sparse_categorical_crossentropy" and output_size == 1:
            error = "label_range"  # label 1 with a single logit (TF InvalidArgument at run time)
        _ = y_last
        return {"one_hot": one_hot, "loss": loss, "metric": metric, "error": error}
    return {"one_hot": False, "loss": "mse", "metric": "mae", "error": None}


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def _act(z: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return np.maximum(z, F32(0))
    if name == "tanh":
        return np.tanh(z)
    if name == "sigmoid":
        return (F32(1) / (F32(1) + np.exp(-z))).astype(F32)
    if name == "softmax":
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True)).astype(F32)
    return z  # linear


def forward(params: list[tuple[np.ndarray, np.ndarray]], X: np.ndarray, hidden: str, output: str,
            masks: list[np.ndarray] | None = None, rate: float = 0.0) -> dict[str, list[np.ndarray]]:
    """Inference (``masks=None``) or training with explicit dropout masks (inverted dropout)."""
    a = np.asarray(X, dtype=F32)
    zs, acts, pre, dropped = [], [a], [], []
    for i, (W, b) in enumerate(params):
        z = (a @ W + b).astype(F32)
        zs.append(z)
        last = i == len(params) - 1
        a = _act(z, output if last else hidden).astype(F32)
        pre.append(a)
        if not last and masks is not None and rate > 0:
            a = (a * masks[i] / F32(1.0 - rate)).astype(F32)
            dropped.append(masks[i])
        acts.append(a)
    return {"z": zs, "a": acts, "pre": pre, "masks": dropped}


def _bce_logits(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    # tf.nn.sigmoid_cross_entropy_with_logits
    return (np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))).astype(F32)


def _cce_logits(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = z.max(axis=1, keepdims=True)
    lse = (np.log(np.exp(z - m).sum(axis=1, keepdims=True)) + m).astype(F32)
    return (-(y * (z - lse)).sum(axis=1)).astype(F32)


def per_sample_loss(loss: str, z_out: np.ndarray, p: np.ndarray, y: np.ndarray, output: str) -> np.ndarray:
    if loss == "binary_crossentropy":
        if output == "sigmoid":
            return _bce_logits(z_out, y).mean(axis=1)
        p = np.clip(p, EPS_KERAS, 1 - EPS_KERAS)
        return (-(y * np.log(p) + (1 - y) * np.log(1 - p))).mean(axis=1).astype(F32)
    if loss == "categorical_crossentropy":
        if output == "softmax":
            return _cce_logits(z_out, y)
        p = p / p.sum(axis=1, keepdims=True)
        p = np.clip(p, EPS_KERAS, 1 - EPS_KERAS)
        return (-(y * np.log(p)).sum(axis=1)).astype(F32)
    return ((p - y) ** 2).mean(axis=1).astype(F32)  # mse


def metric_values(metric: str, p: np.ndarray, y: np.ndarray) -> np.ndarray:
    if metric == "binary_accuracy":
        return ((p > 0.5).astype(F32) == y).astype(F32).mean(axis=1)
    if metric == "categorical_accuracy":
        return (p.argmax(axis=1) == y.argmax(axis=1)).astype(F32)
    return np.abs(p - y).mean(axis=1).astype(F32)  # mae


def _act_grad(a: np.ndarray, z: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return (z > 0).astype(F32)
    if name == "tanh":
        return (F32(1) - a * a).astype(F32)
    if name == "sigmoid":
        return (a * (F32(1) - a)).astype(F32)
    return np.ones_like(a)


def gradients(params, X, y, hidden: str, output: str, loss: str,
              masks: list[np.ndarray] | None = None, rate: float = 0.0):
    """Mean loss over the batch and its gradient for every (W, b)."""
    f = forward(params, X, hidden, output, masks, rate)
    n = X.shape[0]
    z_out, p = f["z"][-1], f["a"][-1]
    loss_value = per_sample_loss(loss, z_out, p, y, output).mean(dtype=F32)
    k = p.shape[1]
    if loss == "binary_crossentropy" and output == "sigmoid":
        dz = (p - y) / F32(n * k)
    elif loss == "categorical_crossentropy" and output == "softmax":
        dz = (p - y) / F32(n)
    else:  # mse through the output activation
        dp = F32(2) * (p - y) / F32(n * k)
        if output == "softmax":
            dz = p * (dp - (dp * p).sum(axis=1, keepdims=True))
        else:
            dz = dp * _act_grad(p, z_out, output)
    dz = dz.astype(F32)
    grads = []
    for i in range(len(params) - 1, -1, -1):
        a_prev = f["a"][i]
        W, _ = params[i]
        grads.append(((a_prev.T @ dz).astype(F32), dz.sum(axis=0).astype(F32)))
        if i > 0:
            da = (dz @ W.T).astype(F32)
            if masks is not None and rate > 0:
                da = (da * masks[i - 1] / F32(1.0 - rate)).astype(F32)
            dz = (da * _act_grad(f["pre"][i - 1], f["z"][i - 1], hidden)).astype(F32)
    grads.reverse()
    return float(loss_value), grads


# ---------------------------------------------------------------------------
# Optimizers (keras/src/optimizers/{adam,sgd,rmsprop}.py, Keras 3.12 defaults)
# ---------------------------------------------------------------------------
@dataclass
class Optimizer:
    name: str
    lr: np.float32
    iterations: int = 0
    slots: list[list[np.ndarray]] = field(default_factory=list)

    def step(self, variables: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if not self.slots:
            self.slots = [[np.zeros_like(v), np.zeros_like(v)] for v in variables]
        lr = F32(self.lr)
        t = F32(self.iterations + 1)
        for v, g, (m, s) in zip(variables, grads, self.slots):
            if self.name == "adam":
                b1, b2 = F32(0.9), F32(0.999)
                alpha = lr * np.sqrt(F32(1) - b2 ** t) / (F32(1) - b1 ** t)
                m += (g - m) * (F32(1) - b1)
                s += (g * g - s) * (F32(1) - b2)
                v -= (m * alpha) / (np.sqrt(s) + F32(1e-7))
            elif self.name == "rmsprop":
                rho = F32(0.9)
                s[...] = rho * s + (F32(1) - rho) * (g * g)
                v -= (lr * g) / np.sqrt(s + F32(1e-7))  # epsilon INSIDE the sqrt (Keras 3)
            else:  # sgd, momentum 0
                v -= g * lr
        self.iterations += 1


def _flat(params):
    out = []
    for W, b in params:
        out.extend([W, b])
    return out


# ---------------------------------------------------------------------------
# model.fit(shuffle=False) with the app's callbacks
# ---------------------------------------------------------------------------
def evaluate(params, X, y, hidden, output, loss, metric, batch_size) -> tuple[float, float]:
    """Keras ``evaluate``/validation: batch-size-weighted means of the loss and the metric."""
    tot_l = tot_m = 0.0
    n = X.shape[0]
    for s in range(0, n, batch_size):
        xb, yb = X[s:s + batch_size], y[s:s + batch_size]
        f = forward(params, xb, hidden, output)
        lb = per_sample_loss(loss, f["z"][-1], f["a"][-1], yb, output).mean(dtype=F32)
        mb = metric_values(metric, f["a"][-1], yb)
        tot_l += float(lb) * len(xb)
        tot_m += float(mb.sum(dtype=F32))
    return tot_l / n, tot_m / n


def metric_key(metric: str) -> str:
    return "mae" if metric == "mae" else "accuracy"


def train(params, X, y, *, hidden, output, loss, metric, optimizer, lr, batch_size, epochs,
          validation_split, early_stopping, patience, reduce_lr, lr_factor,
          record_steps: int = 3) -> dict[str, Any]:
    """Level (a): deterministic ``fit`` without shuffling and without dropout.

    Per batch, Keras computes the predictions once (weights BEFORE the step), takes the loss
    and the metric from them, and then applies the gradient. The epoch's ``loss`` and metric
    are means weighted by the batch size; ``val_*`` use the weights at the end of the epoch.
    Callbacks run in the app's order: EarlyStopping, then ReduceLROnPlateau.
    """
    params = [(np.asarray(W, F32).copy(), np.asarray(b, F32).copy()) for W, b in params]
    X = np.asarray(X, dtype=F32)
    y = np.asarray(y, dtype=F32)
    if y.ndim == 1:
        y = y[:, None]
    split = keras_validation_split(X.shape[0], validation_split)
    Xt, yt, Xv, yv = X[:split], y[:split], X[split:], y[split:]
    opt = Optimizer(optimizer, F32(lr))
    rl_patience = max(1, (patience or 10) // 2)
    es = {"best": None, "best_epoch": 0, "wait": 0, "best_weights": None, "stopped_epoch": 0}
    rl = {"best": None, "wait": 0}
    mk = metric_key(metric)
    history: dict[str, list[float]] = {"loss": [], mk: [], "val_loss": [], "val_" + mk: []}
    if reduce_lr:
        history["learning_rate"] = []
    steps: list[dict] = []
    for epoch in range(epochs):
        tot_l = tot_m = 0.0
        for s in range(0, split, batch_size):
            xb, yb = Xt[s:s + batch_size], yt[s:s + batch_size]
            f = forward(params, xb, hidden, output)
            tot_m += float(metric_values(metric, f["a"][-1], yb).sum(dtype=F32))
            lval, grads = gradients(params, xb, yb, hidden, output, loss)
            flat_v = _flat(params)
            opt.step(flat_v, [g for pair in grads for g in pair])
            params = [(flat_v[2 * i], flat_v[2 * i + 1]) for i in range(len(params))]
            tot_l += lval * len(xb)
            if epoch == 0 and len(steps) < record_steps:
                steps.append({"loss": lval, "params": [(W.copy(), b.copy()) for W, b in params]})
        history["loss"].append(tot_l / split)
        history[mk].append(tot_m / split)
        vl, vm = evaluate(params, Xv, yv, hidden, output, loss, metric, batch_size)
        history["val_loss"].append(vl)
        history["val_" + mk].append(vm)
        if reduce_lr:
            history["learning_rate"].append(float(opt.lr))
        stop = False
        if early_stopping:
            if es["best_weights"] is None:
                es["best_weights"] = [(W.copy(), b.copy()) for W, b in params]
                es["best_epoch"] = epoch
            es["wait"] += 1
            if es["best"] is None or vl < es["best"]:
                es.update(best=vl, best_epoch=epoch, wait=0,
                          best_weights=[(W.copy(), b.copy()) for W, b in params])
            elif es["wait"] >= patience and epoch > 0:
                es["stopped_epoch"] = epoch
                stop = True
        if reduce_lr:
            if rl["best"] is None or vl + 1e-4 < rl["best"]:
                rl.update(best=vl, wait=0)
            else:
                rl["wait"] += 1
                if rl["wait"] >= rl_patience:
                    old = float(opt.lr)
                    if old > F32(1e-7):
                        opt.lr = F32(max(old * lr_factor, 1e-7))
                        rl["wait"] = 0
        if stop:
            break
    if early_stopping and es["best_weights"] is not None:
        params = es["best_weights"]  # Keras 3.12 restores them in on_train_end, always
    return {"params": params, "history": history, "steps": steps,
            "best_epoch": es["best_epoch"] if early_stopping else None,
            "stopped_epoch": es["stopped_epoch"] if early_stopping else None}


# ---------------------------------------------------------------------------
# Views computed on a trained network (viz/nn.py)
# ---------------------------------------------------------------------------
def weight_stats(params) -> dict[str, Any]:
    """🧠 Análisis de Pesos: per-layer and global statistics (np.std = population std)."""
    layers = []
    for W, b in params:
        layers.append({
            "shape": list(W.shape), "mean_w": float(np.mean(W)), "std_w": float(np.std(W)),
            "near_zero_pct": float(round(np.mean(np.abs(W) < 1e-3) * 100, 1)),
            "near_zero_pct_raw": float(np.mean(np.abs(W) < 1e-3) * 100),
            "mean_b": float(np.mean(b)), "std_b": float(np.std(b)), "n_params": int(W.size + b.size),
            "min_w": float(W.min()), "max_w": float(W.max()), "min_b": float(b.min()), "max_b": float(b.max()),
        })
    allw = np.concatenate([W.ravel() for W, _ in params])
    return {"layers": layers, "global_mean": float(np.mean(allw)), "global_std": float(np.std(allw)),
            "global_near_zero_pct": float(np.mean(np.abs(allw) < 1e-3) * 100)}


def activation_stats(params, X_sample, hidden: str, dead_threshold: float, constant_std: float):
    """📉 Análisis de Activaciones on the first ``sample_size`` test rows (hidden layers only)."""
    f = forward(params, X_sample, hidden, hidden)
    out = []
    for acts in f["a"][1:-1]:
        means, stds = acts.mean(axis=0), acts.std(axis=0)
        dead = (acts <= dead_threshold).all(axis=0)
        const = (~dead) & (stds < constant_std)
        bounded = acts.max() <= 1.2
        sat_ratio = float((acts >= 0.99).mean() * 100) if bounded else 0.0
        out.append({"dead": np.where(dead)[0].tolist(), "constant": np.where(const)[0].tolist(),
                    "saturation_pct": sat_ratio, "mean_act": float(means.mean()),
                    "std_act": float(stds.mean()), "neuron_means": means.astype(float).tolist()})
    return out
