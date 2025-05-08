"""Shared setup for the CIFAR-10 adversarial robustness notebooks.

Everything in this repository works on images in ``[0, 1]``.  Normalisation
lives inside the model wrapper (:class:`Normalised`), not in the dataset
transform, so an attack can add its perturbation to raw pixels, clip to the
epsilon ball, and clamp to the valid image box without having to un-normalise
first.  That is the single convention that keeps the epsilon budget meaning
the same thing in every notebook.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

__all__ = [
    "CLASSES", "CIFAR10_MEAN", "CIFAR10_STD", "DEVICE", "SEED",
    "set_seed", "Normalised", "resnet20", "cifar10", "Penultimate",
    "fgsm", "pgd", "mifgsm", "apgd", "cw_margin",
    "accuracy", "collect_features", "plot_umap", "show_attack_grid",
]

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# Per-channel statistics of the CIFAR-10 training set.
#
# Note the std: (0.2023, 0.1994, 0.2010) is very widely copied but wrong — it
# is the std of the *mean-subtracted, per-image normalised* data from an old
# Torch7 recipe.  Measured on the raw training images the values are the ones
# below.  chenyaofo/pytorch-cifar-models, whose weights are used here, was
# trained with these.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Where torchvision should keep (or find) the CIFAR-10 archive.
DATA_DIR = os.environ.get("CIFAR10_DATA_DIR", "data")


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and torch. Call at the top of every notebook."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------- model


class Normalised(nn.Module):
    """Wrap a CIFAR-10 classifier so that it accepts images in ``[0, 1]``."""

    def __init__(self, model: nn.Module, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)

    @property
    def head(self) -> nn.Module:
        """The final linear layer of the wrapped network."""
        return self.model.fc


def resnet20(pretrained: bool = True) -> Normalised:
    """ResNet-20 for CIFAR-10, wrapped to take ``[0, 1]`` input.

    ``pretrained=True`` gives the chenyaofo checkpoint, 92.59% clean test
    accuracy. Every experiment here starts from it.
    """
    net = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet20",
        pretrained=pretrained, verbose=False,
    )
    return Normalised(net).to(DEVICE)


class Penultimate:
    """Capture the input of the classifier head on every forward pass.

    A forward pre-hook on ``fc`` gives the penultimate features without
    replacing the head, so the same model still produces logits.

        feats = Penultimate(model)
        logits = model(x)
        feats.value          # (N, 64)
    """

    def __init__(self, model: Normalised):
        self.value: torch.Tensor | None = None
        self._handle = model.head.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        self.value = inputs[0].detach()

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False


# ---------------------------------------------------------------------- data

_AUGMENT = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
]


def cifar10(
    train: bool = False,
    augment: bool = False,
    batch_size: int = 128,
    fraction: float = 1.0,
    shuffle: bool | None = None,
    seed: int = SEED,
) -> DataLoader:
    """A CIFAR-10 loader yielding images in ``[0, 1]`` — never normalised.

    ``fraction`` takes a deterministic random subset, which is how the
    fine-tuning experiments keep their epochs short.
    """
    steps = (_AUGMENT if augment else []) + [transforms.ToTensor()]
    ds = datasets.CIFAR10(
        DATA_DIR, train=train, download=True,
        transform=transforms.Compose(steps),
    )
    if fraction < 1.0:
        g = np.random.default_rng(seed)
        keep = g.permutation(len(ds))[: int(len(ds) * fraction)]
        ds = Subset(ds, keep.tolist())
    if shuffle is None:
        shuffle = train
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ------------------------------------------------------------------- attacks
#
# Every attack takes and returns images in [0, 1] and guarantees
#   ||x_adv - x||_inf <= eps   and   x_adv in [0, 1]
# for the eps it was given. Defaults are 8/255, the standard CIFAR-10 budget.


def _project(x_adv, x, eps):
    """Clip to the eps-ball around x, then to the valid image box."""
    return torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)


def cw_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Carlini-Wagner margin, per sample: ``max_{i != y} z_i - z_y``.

    Positive means the sample is already misclassified, so an attack wants to
    maximise it.  Masking with ``-inf`` rather than multiplying the one-hot
    out matters: with ``(1 - onehot) * logits`` the zeroed true-class slot
    wins ``max`` whenever every other logit is negative.
    """
    onehot = F.one_hot(y, logits.size(1)).bool()
    correct = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.masked_fill(onehot, float("-inf")).max(1).values
    return other - correct


def _loss_and_grad(model, x, y, loss):
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    f = (F.cross_entropy(logits, y, reduction="none") if loss == "ce"
         else cw_margin(logits, y))
    grad = torch.autograd.grad(f.sum(), x)[0]
    return f.detach(), grad


def fgsm(model, x, y, eps: float = 8 / 255):
    """Goodfellow et al. 2015 — one signed-gradient step."""
    model.eval()
    _, grad = _loss_and_grad(model, x, y, "ce")
    return _project(x + eps * grad.sign(), x, eps).detach()


def pgd(model, x, y, eps: float = 8 / 255, alpha: float = 2 / 255,
        steps: int = 10, random_start: bool = True):
    """Madry et al. 2018 — projected gradient descent, L-inf."""
    model.eval()
    x_adv = x.clone().detach()
    if random_start:
        x_adv = _project(x_adv + torch.empty_like(x).uniform_(-eps, eps), x, eps)
    for _ in range(steps):
        _, grad = _loss_and_grad(model, x_adv, y, "ce")
        x_adv = _project(x_adv + alpha * grad.sign(), x, eps).detach()
    return x_adv


def mifgsm(model, x, y, eps: float = 8 / 255, alpha: float = 2 / 255,
           steps: int = 10, mu: float = 1.0):
    """Dong et al. 2018 — PGD with a momentum term.

    The gradient is divided by its own L1 norm before accumulating, which is
    what keeps ``mu`` meaningful across steps whose gradients differ in scale.
    Accumulating ``grad.sign()`` instead would make the decay the only thing
    the momentum does.
    """
    model.eval()
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        _, grad = _loss_and_grad(model, x_adv, y, "ce")
        grad = grad / grad.abs().flatten(1).sum(1).view(-1, 1, 1, 1).clamp_min(1e-12)
        momentum = mu * momentum + grad
        x_adv = _project(x_adv + alpha * momentum.sign(), x, eps).detach()
    return x_adv


def _apgd_checkpoints(steps: int) -> list[int]:
    """The ``w_j`` of Croce & Hein 2020: p0=0, p1=0.22, growing gaps."""
    p = [0.0, 0.22]
    while True:
        nxt = p[-1] + max(p[-1] - p[-2] - 0.03, 0.06)
        if nxt > 1.0:
            break
        p.append(nxt)
    return sorted({int(np.ceil(pi * steps)) for pi in p[1:]})


def apgd(model, x, y, eps: float = 8 / 255, steps: int = 20,
         loss: str = "ce", rho: float = 0.75, momentum: float = 0.75):
    """Croce & Hein 2020 — Auto-PGD, L-inf.

    The three things that make it Auto-PGD rather than PGD with a different
    name, all of which the step-size-free claim depends on:

    * the step size starts at ``2 * eps`` and is **halved at checkpoints**
      where progress stalled, rather than being a constant the caller tunes;
    * updates carry **momentum** over the previous iterate;
    * on halving, the search **restarts from the best point found so far**
      rather than continuing from wherever it had drifted to.
    """
    model.eval()
    x, y = x.to(DEVICE), y.to(DEVICE)
    n = x.size(0)
    ckpts = set(_apgd_checkpoints(steps))

    eta = torch.full((n, 1, 1, 1), 2.0 * eps, device=x.device)
    x_adv = _project(x + torch.empty_like(x).uniform_(-eps, eps), x, eps)

    f, grad = _loss_and_grad(model, x_adv, y, loss)
    x_best, f_best = x_adv.clone(), f.clone()

    x_prev = x_adv
    x_adv = _project(x_adv + eta * grad.sign(), x, eps)

    improved_count = torch.zeros(n, device=x.device)
    eta_at_ckpt, f_at_ckpt, last_ckpt = eta.clone(), f_best.clone(), 0

    for k in range(1, steps):
        f, grad = _loss_and_grad(model, x_adv, y, loss)

        better = f > f_best
        improved_count += better.float()
        x_best = torch.where(better.view(-1, 1, 1, 1), x_adv, x_best)
        f_best = torch.where(better, f, f_best)

        z = _project(x_adv + eta * grad.sign(), x, eps)
        step = momentum * (z - x_adv) + (1.0 - momentum) * (x_adv - x_prev)
        x_prev, x_adv = x_adv, _project(x_adv + step, x, eps)

        if k in ckpts:
            stalled = improved_count < rho * (k - last_ckpt)
            flat = (eta_at_ckpt.view(-1) == eta.view(-1)) & (f_at_ckpt == f_best)
            halve = (stalled | flat).view(-1, 1, 1, 1)
            eta = torch.where(halve, eta / 2.0, eta)
            x_adv = torch.where(halve, x_best, x_adv)
            x_prev = x_adv.clone()
            eta_at_ckpt, f_at_ckpt = eta.clone(), f_best.clone()
            improved_count.zero_()
            last_ckpt = k

    with torch.no_grad():
        logits = model(x_adv)
        f = (F.cross_entropy(logits, y, reduction="none") if loss == "ce"
             else cw_margin(logits, y))
    better = f > f_best
    x_best = torch.where(better.view(-1, 1, 1, 1), x_adv, x_best)
    return x_best.detach()


# ---------------------------------------------------------------- evaluation


def accuracy(model, loader, attack=None, **attack_kwargs) -> float:
    """Top-1 accuracy in percent, optionally under ``attack``."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if attack is not None:
            x = attack(model, x, y, **attack_kwargs)
        with torch.no_grad():
            correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def collect_features(model, loader, n_samples: int = 1000,
                     attacks: dict | None = None, **attack_kwargs):
    """Penultimate features for clean and attacked copies of the same images.

    ``attacks`` maps a condition name to an attack function; ``None`` as the
    value means "leave the images alone".  Returns ``(features, labels,
    conditions)`` as three aligned arrays, ready for :func:`plot_umap`.
    """
    attacks = attacks or {"Clean": None, "FGSM": fgsm, "PGD": pgd}
    model.eval()
    feats, labels, conds = [], [], []
    probe = Penultimate(model)
    try:
        seen = 0
        for x, y in loader:
            if seen >= n_samples:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            for name, attack in attacks.items():
                xa = x if attack is None else attack(model, x, y, **attack_kwargs)
                with torch.no_grad():
                    model(xa)
                feats.append(probe.value.cpu().numpy())
                labels.append(y.cpu().numpy())
                conds += [name] * y.numel()
            seen += y.numel()
    finally:
        probe.remove()
    return np.concatenate(feats), np.concatenate(labels), np.array(conds)


# ------------------------------------------------------------------ plotting

_MARKERS = {"Clean": "o", "FGSM": "^", "PGD": "s"}


def plot_umap(features, labels, conditions, title: str, path=None,
              n_neighbors: int = 10, min_dist: float = 0.1, seed: int = SEED):
    """UMAP of the penultimate space, marker by condition, colour by class."""
    import matplotlib.pyplot as plt
    import umap

    emb = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed,
    ).fit_transform(features)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cond in dict.fromkeys(conditions):
        m = conditions == cond
        ax.scatter(
            emb[m, 0], emb[m, 1], c=labels[m], cmap="tab10", vmin=0, vmax=9,
            marker=_MARKERS.get(cond, "o"), s=14, alpha=0.7, linewidths=0,
            label=cond,
        )
    handles = [
        plt.Line2D([], [], marker=_MARKERS.get(c, "o"), ls="", color="0.3",
                   label=c)
        for c in dict.fromkeys(conditions)
    ]
    handles += [
        plt.Line2D([], [], marker="o", ls="", color=plt.cm.tab10(i / 9),
                   label=name)
        for i, name in enumerate(CLASSES)
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False)
    ax.set(title=title, xticks=[], yticks=[])
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def show_attack_grid(model, x, y, attack, title: str, n: int = 8, path=None,
                     **attack_kwargs):
    """Original / perturbation / adversarial, with true and predicted names."""
    import matplotlib.pyplot as plt

    model.eval()
    x, y = x[:n].to(DEVICE), y[:n].to(DEVICE)
    x_adv = attack(model, x, y, **attack_kwargs)
    with torch.no_grad():
        p_clean = model(x).argmax(1)
        p_adv = model(x_adv).argmax(1)

    delta = (x_adv - x).cpu()
    rows = [
        (x.cpu(), "original", p_clean.cpu()),
        # Perturbations are tiny by construction, so show them rescaled to
        # [0, 1] rather than as they actually are — otherwise every panel is
        # uniform grey.
        ((delta - delta.amin()) / (delta.amax() - delta.amin() + 1e-12),
         "perturbation", None),
        (x_adv.cpu(), "adversarial", p_adv.cpu()),
    ]

    fig, axes = plt.subplots(3, n, figsize=(1.5 * n, 5.2))
    for r, (imgs, name, preds) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c].permute(1, 2, 0).numpy())
            ax.set_xticks([]), ax.set_yticks([])
            if preds is not None:
                ok = preds[c].item() == y[c].item()
                ax.set_title(CLASSES[preds[c]], fontsize=8,
                             color="black" if ok else "crimson")
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.suptitle(f"{title} — true labels: "
                 + ", ".join(CLASSES[i] for i in y.cpu()), fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
