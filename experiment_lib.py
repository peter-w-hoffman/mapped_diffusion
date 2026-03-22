# File: experiment_lib.py

import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.utils import make_grid

from common import device
from ddpm_lib import base_m11_transform, forward_backward_reconstruct
from classifier_lib import classify_raw_m11


def get_raw_mnist_dataset(root="./data", split="test"):
    train = split == "train"
    return datasets.MNIST(root=root, train=train, download=True, transform=base_m11_transform())


def sample_subset_batch(subset, n_samples, root="./data", split="test", seed=2026, device=device):
    """
    Uniform over all images whose labels lie in subset.
    Returned x is in raw DDPM image space [-1,1].
    """
    subset = sorted(list(subset))
    ds = get_raw_mnist_dataset(root=root, split=split)

    targets = ds.targets
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for c in subset:
        mask |= (targets == int(c))

    idx = torch.where(mask)[0]
    g = torch.Generator()
    g.manual_seed(int(seed))
    picked = idx[torch.randperm(len(idx), generator=g)[:n_samples]]

    x_list = [ds[int(i)][0] for i in picked.tolist()]
    x_raw = torch.stack(x_list, dim=0).to(device)
    y = targets[picked].to(device)
    return x_raw, y


@torch.no_grad()
def evaluate_clean_subset_rate(clf_model, clf_cfg, x_raw_m11, subset):
    """
    Classifier performance on the clean images before any diffusion.
    Reported with t = -1 so it stays separate from DDPM time indexing.
    """
    subset_tensor = torch.tensor(sorted(list(subset)), device=x_raw_m11.device)

    clf_out = classify_raw_m11(
        clf_model,
        x_raw_m11,
        input_space=clf_cfg["input_space"],
        mnist_mean=clf_cfg["mnist_mean"],
        mnist_std=clf_cfg["mnist_std"],
    )

    within = torch.isin(clf_out["preds"], subset_tensor)
    subset_prob = clf_out["probs"][:, subset_tensor].sum(dim=1)

    return {
        "t": -1,
        "within_subset_rate": within.float().mean().item(),
        "mean_pred_confidence": clf_out["confidences"].mean().item(),
        "mean_subset_probability": subset_prob.mean().item(),
    }


@torch.no_grad()
def run_time_sweep_on_fixed_batch(
    ddpm_model,
    schedule,
    precond,
    clf_model,
    clf_cfg,
    x_raw_m11,
    y_true,
    subset,
    time_steps,
    batch_size=128,
    forward_seed_base=12345,
    reverse_seed_base=54321,
    store_time_steps=(),
    n_store=25,
):
    """
    Runs the main experiment on a fixed batch of original images so that
    different preconditioners can be compared on exactly the same inputs.
    """
    ddpm_model.eval()
    clf_model.eval()

    subset = sorted(list(subset))
    subset_tensor = torch.tensor(subset, device=x_raw_m11.device)

    results = []
    examples = {}

    for t in time_steps:
        preds_all = []
        conf_all = []
        subset_prob_all = []

        store_x0 = []
        store_xrec = []
        store_preds = []
        store_conf = []
        store_y = []

        for start in range(0, x_raw_m11.size(0), batch_size):
            end = min(start + batch_size, x_raw_m11.size(0))
            xb = x_raw_m11[start:end]
            yb = y_true[start:end]

            fb = forward_backward_reconstruct(
                ddpm_model,
                schedule,
                precond,
                xb,
                t_start=int(t),
                forward_seed=int(forward_seed_base + 100000 * int(t) + start),
                reverse_seed=int(reverse_seed_base + 100000 * int(t) + start),
            )

            clf_out = classify_raw_m11(
                clf_model,
                fb["x_rec_raw"],
                input_space=clf_cfg["input_space"],
                mnist_mean=clf_cfg["mnist_mean"],
                mnist_std=clf_cfg["mnist_std"],
            )

            preds_all.append(clf_out["preds"].detach())
            conf_all.append(clf_out["confidences"].detach())
            subset_prob_all.append(clf_out["probs"][:, subset_tensor].sum(dim=1).detach())

            if t in store_time_steps and sum(x.size(0) for x in store_x0) < n_store:
                keep = min(n_store - sum(x.size(0) for x in store_x0), xb.size(0))
                store_x0.append(xb[:keep].detach().cpu())
                store_xrec.append(fb["x_rec_raw"][:keep].detach().cpu())
                store_preds.append(clf_out["preds"][:keep].detach().cpu())
                store_conf.append(clf_out["confidences"][:keep].detach().cpu())
                store_y.append(yb[:keep].detach().cpu())

        preds = torch.cat(preds_all, dim=0)
        confs = torch.cat(conf_all, dim=0)
        subset_probs = torch.cat(subset_prob_all, dim=0)

        within = torch.isin(preds, subset_tensor)
        row = {
            "t": int(t),
            "within_subset_rate": within.float().mean().item(),
            "mean_pred_confidence": confs.mean().item(),
            "mean_subset_probability": subset_probs.mean().item(),
        }
        results.append(row)

        if t in store_time_steps:
            examples[int(t)] = {
                "x0_raw": torch.cat(store_x0, dim=0),
                "x_rec_raw": torch.cat(store_xrec, dim=0),
                "preds": torch.cat(store_preds, dim=0),
                "confidences": torch.cat(store_conf, dim=0),
                "y_true": torch.cat(store_y, dim=0),
            }

    return {
        "results": results,
        "examples": examples,
        "subset": subset,
        "sampled_labels": y_true.detach().cpu(),
    }


def plot_time_sweep(results, title=None, label=None, ax=None, as_bar=False):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ts = [row["t"] for row in results]
    ys = [row["within_subset_rate"] for row in results]

    if as_bar:
        ax.bar(ts, ys, alpha=0.85, label=label)
    else:
        ax.plot(ts, ys, marker="o", linewidth=2, label=label)

    ax.set_xlabel("diffusion time step")
    ax.set_ylabel("proportion predicted inside subset")
    ax.set_ylim(0.0, 1.0)

    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend()

    return ax


def plot_mode_comparison(results_by_name, subset, as_bar=False, baseline=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, results in results_by_name.items():
        plot_time_sweep(results, label=name, ax=ax, as_bar=as_bar)
    if baseline is not None:
        ax.axhline(baseline["within_subset_rate"], linestyle="--", linewidth=1.5, label="clean baseline")
        ax.legend()
    ax.set_title(f"Within-subset rate for subset {sorted(list(subset))}")
    plt.show()


def show_reconstruction_examples(experiment_output, t, nrow=5):
    pack = experiment_output["examples"][int(t)]
    x0 = (pack["x0_raw"].clamp(-1, 1) + 1.0) / 2.0
    xrec = (pack["x_rec_raw"].clamp(-1, 1) + 1.0) / 2.0

    grid0 = make_grid(x0, nrow=nrow, padding=6)
    grid1 = make_grid(xrec, nrow=nrow, padding=6)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("original")
    plt.imshow(grid0.permute(1, 2, 0).squeeze(), cmap="gray")

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title(f"reconstructed @ t={t}")
    plt.imshow(grid1.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.show()

    print("y_true:", pack["y_true"].tolist())
    print("preds :", pack["preds"].tolist())
    print("conf  :", [round(x, 4) for x in pack["confidences"].tolist()])