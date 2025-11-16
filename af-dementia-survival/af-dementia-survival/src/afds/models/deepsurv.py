
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..metrics import c_index


class DeepSurv(nn.Module):
    """Minimal DeepSurv-style feedforward network producing a single log-risk (linear predictor)."""

    def __init__(self, in_dim: int, hidden: Iterable[int] = (64, 32), dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x).squeeze(-1)


def neg_partial_lik(lp: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Negative log partial likelihood for Cox models (Breslow)."""
    order = torch.argsort(durations, descending=False)
    t = durations[order]
    e = events[order].float()
    r = torch.exp(lp[order])

    rev_csum = torch.cumsum(r.flip(0), dim=0).flip(0)
    uniq_t, first_idx = torch.unique_consecutive(t, return_counts=False, return_inverse=False, return_indices=True)
    denom = rev_csum[first_idx]
    d_j = torch.stack([e[first_idx[i] : (first_idx[i + 1] if i + 1 < len(first_idx) else len(e))].sum() for i in range(len(first_idx))])

    mask = d_j > 0
    return -((lp * e).sum() - torch.sum(torch.log(denom[mask]) * d_j[mask]))


def fit_deepsurv_hpo(
    X_train: np.ndarray,
    ytime_train: np.ndarray,
    yevent_train: np.ndarray,
    X_val: np.ndarray,
    ytime_val: np.ndarray,
    yevent_val: np.ndarray,
    search_space: Optional[Dict[str, Iterable[Any]]] = None,
    n_trials: int = 20,
    random_state: int = 42,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Random-search HPO for DeepSurv; returns best model + params."""
    rng = np.random.RandomState(random_state)
    space = search_space or {
        "hidden": [(64, 32), (128, 64), (128, 64, 32)],
        "dropout": [0.0, 0.1, 0.2],
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "epochs": [30, 50, 80],
        "batch_size": [128, 256, 512],
    }

    def sample() -> Dict[str, Any]:
        return {k: rng.choice(v) for k, v in space.items()}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr_t = torch.tensor(ytime_train, dtype=torch.float32).to(device)
    ytr_e = torch.tensor(yevent_train, dtype=torch.float32).to(device)

    Xva = torch.tensor(X_val, dtype=torch.float32).to(device)
    yva_t = torch.tensor(ytime_val, dtype=torch.float32).to(device)
    yva_e = torch.tensor(yevent_val, dtype=torch.float32).to(device)

    best: Dict[str, Any] = {"cindex": -1.0, "params": None, "model": None, "state": None}

    for _ in range(int(n_trials)):
        hp = sample()
        model = DeepSurv(Xtr.shape[1], hidden=hp["hidden"], dropout=hp["dropout"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

        bs = int(hp["batch_size"])
        epochs = int(hp["epochs"])

        for _ep in range(epochs):
            model.train()
            idx = np.random.permutation(len(Xtr))
            for i in range(0, len(Xtr), bs):
                j = idx[i : i + bs]
                opt.zero_grad()
                lp = model(Xtr[j])
                loss = neg_partial_lik(lp, ytr_t[j], ytr_e[j])
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            lp_val = model(Xva).detach().cpu().numpy()
        ci = c_index(ytime_val, yevent_val, lp_val)

        if ci > best["cindex"]:
            best = {"cindex": float(ci), "params": hp, "model": model, "state": model.state_dict()}

    return best


def predict_lp(model: DeepSurv, X: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    """Predict linear predictor with DeepSurv model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).to(device)
        lp = model(X).detach().cpu().numpy()
    return lp
