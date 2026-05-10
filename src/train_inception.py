from __future__ import annotations

import json
import os

import numpy as np

from .artifacts import ensure_dirs, fit_or_load_scaler, get_or_create_split_indices
from .config import INCEPTION_MODEL_PATH, N_TIMESTEPS, REPORTS_DIR, SEED
from .data import load_ecg5000
from .metrics import multiclass_metrics, save_json, save_multiclass_confusion_matrix


def main(epochs: int = 20, lr: float = 3e-3, batch_size: int = 64) -> None:
    """Train InceptionTime with tsai/fastai.

    This script is ready for Colab MCP or a local environment with fastai/tsai.
    It intentionally uses only the inner training fold for scaler fitting and
    validation for model selection.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        import torch
        from fastai.learner import Learner
        from fastai.losses import CrossEntropyLossFlat
        from fastai.metrics import accuracy
        from fastai.optimizer import ranger
        from fastai.torch_core import set_seed
        from tsai.all import CategoryBlock, DataBlock, InceptionTime, ItemGetter, L, TSTensorBlock, itemify
    except ImportError as exc:
        raise SystemExit(
            "fastai/tsai are required for InceptionTime training. "
            "Use Colab MCP or install the optional deep-learning dependencies first. "
            f"Original error: {exc}"
        ) from exc

    ensure_dirs()
    set_seed(SEED, reproducible=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    X_full, y_full = load_ecg5000("train")
    split = get_or_create_split_indices(y_full)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    scaler = fit_or_load_scaler(X_full, train_idx)

    X_train = scaler.transform(X_full[train_idx])
    X_val = scaler.transform(X_full[val_idx])
    y_train = y_full[train_idx].astype(str)
    y_val = y_full[val_idx].astype(str)

    X_all = np.concatenate([X_train, X_val], axis=0)
    X_all = X_all.reshape(X_all.shape[0], N_TIMESTEPS, 1).transpose(0, 2, 1)
    y_all = np.concatenate([y_train, y_val], axis=0)
    splits = (
        L(np.arange(0, len(y_train)), use_list=True),
        L(np.arange(len(y_train), len(y_all)), use_list=True),
    )

    dblock = DataBlock(
        blocks=(TSTensorBlock, CategoryBlock),
        getters=[ItemGetter(0), ItemGetter(1)],
        splitter=lambda _: splits,
    )
    dls = dblock.dataloaders(itemify(X_all, y_all), bs=batch_size, val_bs=batch_size * 2)
    net = InceptionTime(dls.dataset[0][0].shape[-2], dls.c)
    learn = Learner(dls, net, loss_func=CrossEntropyLossFlat(), metrics=accuracy, opt_func=ranger)
    learn.fit_flat_cos(epochs, lr)

    preds, targets = learn.get_preds(dl=dls.valid)
    vocab = [int(str(v)) for v in dls.vocab]
    pred_labels = np.asarray([vocab[i] for i in preds.argmax(dim=1).cpu().numpy()])
    true_labels = np.asarray([vocab[i] for i in targets.cpu().numpy()])

    learn.export(INCEPTION_MODEL_PATH)
    metrics = multiclass_metrics(true_labels, pred_labels)
    metrics["metadata"] = {"epochs": epochs, "lr": lr, "batch_size": batch_size, "seed": SEED}
    save_json(metrics, REPORTS_DIR / "metrics_inception_validation.json")
    save_multiclass_confusion_matrix(
        true_labels,
        pred_labels,
        REPORTS_DIR / "confusion_matrix_inception_validation.png",
        "InceptionTime validation confusion matrix",
    )
    with (REPORTS_DIR / "inception_training_config.json").open("w", encoding="utf-8") as f:
        json.dump(metrics["metadata"], f, indent=2)
    print(f"Saved InceptionTime learner to {INCEPTION_MODEL_PATH}")
    print(f"Validation macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
