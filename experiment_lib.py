import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.utils import make_grid

from common import device
from ddpm_lib import (
    base_m11_transform,
    forward_backward_reconstruct,
    forward_to_xt,
    reconstruct_from_xt_with_trajectory,
)
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


def sample_uniform_by_class(
    classes,
    n_per_class,
    root="./data",
    split="test",
    seed=2026,
    device=device,
    replace=False,
):
    """
    Sample the same number of images from each requested MNIST class.

    Returned tensors are class-major: all samples for classes[0], then classes[1], etc.
    This makes later per-class slicing simple and stable.
    """
    classes = [int(c) for c in classes]
    n_per_class = int(n_per_class)

    ds = get_raw_mnist_dataset(root=root, split=split)
    targets = ds.targets

    g = torch.Generator()
    g.manual_seed(int(seed))

    x_list = []
    y_list = []
    idx_list = []

    for c in classes:
        idx_c = torch.where(targets == c)[0]
        if idx_c.numel() == 0:
            raise ValueError(f"No samples found for class {c} in split={split!r}")

        if replace or n_per_class > idx_c.numel():
            draw = torch.randint(low=0, high=idx_c.numel(), size=(n_per_class,), generator=g)
            picked = idx_c[draw]
        else:
            perm = torch.randperm(idx_c.numel(), generator=g)[:n_per_class]
            picked = idx_c[perm]

        x_list.extend(ds[int(i)][0] for i in picked.tolist())
        y_list.append(torch.full((n_per_class,), c, dtype=targets.dtype))
        idx_list.append(picked)

    x_raw = torch.stack(x_list, dim=0).to(device)
    y = torch.cat(y_list, dim=0).to(device)
    dataset_indices = torch.cat(idx_list, dim=0)

    return x_raw, y, dataset_indices


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


def show_reconstruction_examples(mode_name, experiment_output, t, nrow=5):
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
    plt.title(f"reconstructed @ t={t} using {mode_name}")
    plt.imshow(grid1.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.show()

    print("y_true:", pack["y_true"].tolist())
    print("preds :", pack["preds"].tolist())
    print("conf  :", [round(x, 4) for x in pack["confidences"].tolist()])


def _ensure_cpu_long(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().long()
    return torch.tensor(x, dtype=torch.long)


def _class_positions_from_labels(y_true):
    y_cpu = _ensure_cpu_long(y_true)
    return {
        int(c): torch.where(y_cpu == int(c))[0].tolist()
        for c in torch.unique(y_cpu).tolist()
    }


def _compute_joint_count_tensor(y_true, left_preds, right_preds, num_classes=10):
    y_true = _ensure_cpu_long(y_true)
    left_preds = _ensure_cpu_long(left_preds)
    right_preds = _ensure_cpu_long(right_preds)

    n_samples, n_times = left_preds.shape
    counts = torch.zeros(num_classes, n_times, num_classes, num_classes, dtype=torch.long)
    class_count_by_time = torch.zeros(num_classes, n_times, dtype=torch.long)

    for i in range(num_classes):
        mask_i = (y_true == i)
        n_i = int(mask_i.sum().item())
        if n_i == 0:
            continue

        class_count_by_time[i].fill_(n_i)
        left_i = left_preds[mask_i]
        right_i = right_preds[mask_i]

        for t_idx in range(n_times):
            flat = left_i[:, t_idx] * num_classes + right_i[:, t_idx]
            binc = torch.bincount(flat, minlength=num_classes * num_classes)
            counts[i, t_idx] = binc.view(num_classes, num_classes)

    return counts, class_count_by_time


def _trajectory_model_to_raw(precond, trajectory_model):
    """
    trajectory_model: [B, L, 1, 28, 28] in DDPM model space
    returns:          [B, L, 1, 28, 28] in raw [-1,1] space
    """
    b, l, c, h, w = trajectory_model.shape
    flat = trajectory_model.reshape(b * l, c, h, w)
    raw = precond.undo(flat).clamp(-1, 1)
    return raw.reshape(b, l, c, h, w)


@torch.no_grad()
def run_joint_label_probability_experiment(
    model_bundles,
    clf_model,
    clf_cfg,
    x_raw_m11,
    y_true,
    time_steps,
    pair_mode_names=("identity", "M"),
    batch_size=128,
    forward_seed_base=1000,
    reverse_seed_base=2000,
    num_classes=10,
    dataset_indices=None,
):
    """
    Core forward-backward experiment.

    For each clean image x0 with true label i and each time t, we run the two DDPM
    modes in `pair_mode_names`, classify both reconstructions, and store:
        (true label i, time t, pred_left, pred_right)

    The main tensor output is:
        joint_counts[i, t_idx, j, j']
    where j is the classifier label from pair_mode_names[0] and j' is the label
    from pair_mode_names[1].

    joint_probs is the corresponding conditional probability tensor
    P(pred_left=j, pred_right=j' | true=i, time=t).
    """
    left_mode, right_mode = pair_mode_names
    for mode_name in pair_mode_names:
        if mode_name not in model_bundles:
            raise KeyError(f"Missing model bundle for mode {mode_name!r}")

    n_samples = x_raw_m11.size(0)
    n_times = len(time_steps)

    preds_by_mode = {
        left_mode: torch.empty((n_samples, n_times), dtype=torch.long),
        right_mode: torch.empty((n_samples, n_times), dtype=torch.long),
    }
    confidences_by_mode = {
        left_mode: torch.empty((n_samples, n_times), dtype=torch.float32),
        right_mode: torch.empty((n_samples, n_times), dtype=torch.float32),
    }

    for t_idx, t in enumerate(time_steps):
        t = int(t)

        for mode_name in pair_mode_names:
            schedule_T = int(model_bundles[mode_name]["schedule"].T)
            if not (0 <= t < schedule_T):
                raise ValueError(f"time step t={t} is outside schedule for mode {mode_name!r} with T={schedule_T}")

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xb = x_raw_m11[start:end]

            forward_seed = int(forward_seed_base + 100000 * t + start)
            reverse_seed = int(reverse_seed_base + 100000 * t + start)

            for mode_name in pair_mode_names:
                bundle = model_bundles[mode_name]
                fb = forward_backward_reconstruct(
                    model=bundle["model"],
                    schedule=bundle["schedule"],
                    precond=bundle["precond"],
                    x_raw_m11=xb,
                    t_start=t,
                    forward_seed=forward_seed,
                    reverse_seed=reverse_seed,
                )

                clf_out = classify_raw_m11(
                    clf_model,
                    fb["x_rec_raw"],
                    input_space=clf_cfg["input_space"],
                    mnist_mean=clf_cfg["mnist_mean"],
                    mnist_std=clf_cfg["mnist_std"],
                )

                preds_by_mode[mode_name][start:end, t_idx] = clf_out["preds"].detach().cpu()
                confidences_by_mode[mode_name][start:end, t_idx] = clf_out["confidences"].detach().cpu()

    y_cpu = _ensure_cpu_long(y_true)
    time_tensor = torch.tensor([int(t) for t in time_steps], dtype=torch.long)

    joint_counts, class_count_by_time = _compute_joint_count_tensor(
        y_true=y_cpu,
        left_preds=preds_by_mode[left_mode],
        right_preds=preds_by_mode[right_mode],
        num_classes=num_classes,
    )

    denom = class_count_by_time.clamp_min(1).unsqueeze(-1).unsqueeze(-1).float()
    joint_probs = joint_counts.float() / denom
    joint_probs = joint_probs.masked_fill(class_count_by_time.eq(0).unsqueeze(-1).unsqueeze(-1), 0.0)

    left_marginal_probs = joint_probs.sum(dim=3)
    right_marginal_probs = joint_probs.sum(dim=2)

    core_tuples = torch.empty((n_samples, n_times, 4), dtype=torch.long)
    core_tuples[:, :, 0] = y_cpu.unsqueeze(1)
    core_tuples[:, :, 1] = time_tensor.unsqueeze(0)
    core_tuples[:, :, 2] = preds_by_mode[left_mode]
    core_tuples[:, :, 3] = preds_by_mode[right_mode]

    return {
        "mode_pair": tuple(pair_mode_names),
        "time_steps": [int(t) for t in time_steps],
        "x0_raw": x_raw_m11.detach().cpu(),
        "y_true": y_cpu,
        "dataset_indices": _ensure_cpu_long(dataset_indices),
        "core_tuples": core_tuples,
        "core_tuple_fields": ["true_label", "time_step", f"pred_{left_mode}", f"pred_{right_mode}"],
        "preds_by_mode": preds_by_mode,
        "confidences_by_mode": confidences_by_mode,
        "joint_counts": joint_counts,
        "joint_probs": joint_probs,
        "class_count_by_time": class_count_by_time,
        "left_marginal_probs": left_marginal_probs,
        "right_marginal_probs": right_marginal_probs,
        "agreement_rate_by_time": (preds_by_mode[left_mode] == preds_by_mode[right_mode]).float().mean(dim=0),
        "class_positions": _class_positions_from_labels(y_cpu),
    }


@torch.no_grad()
def collect_reverse_trajectory_bank(
    model_bundles,
    x_raw_m11,
    y_true,
    time_steps,
    mode_names=None,
    batch_size=64,
    forward_seed_base=3000,
    reverse_seed_base=4000,
    dataset_indices=None,
):
    """
    Save reverse trajectories for a fixed anchor set of clean images.

    Output convention for each mode/time pair:
      reverse_traj_raw[:, 0]     = starting x_t mapped back to raw [-1,1] space
      reverse_traj_raw[:, r + 1] = state after reverse update at reverse_update_steps[r]

    This makes it easy to compare how the reverse process evolves later on.
    """
    if mode_names is None:
        mode_names = list(model_bundles.keys())

    y_cpu = _ensure_cpu_long(y_true)

    bank = {
        "time_steps": [int(t) for t in time_steps],
        "mode_names": list(mode_names),
        "x0_raw": x_raw_m11.detach().cpu(),
        "y_true": y_cpu,
        "dataset_indices": _ensure_cpu_long(dataset_indices),
        "class_positions": _class_positions_from_labels(y_cpu),
        "trajectory_layout": (
            "reverse_traj_raw[:, 0] is the starting x_t in raw space; "
            "reverse_traj_raw[:, r + 1] is the state after reverse update "
            "at reverse_update_steps[r]."
        ),
        "trajectories": {},
    }

    n_samples = x_raw_m11.size(0)

    for mode_name in mode_names:
        if mode_name not in model_bundles:
            raise KeyError(f"Missing model bundle for mode {mode_name!r}")

        bundle = model_bundles[mode_name]
        mode_store = {}

        for t in time_steps:
            t = int(t)
            if not (0 <= t < int(bundle["schedule"].T)):
                raise ValueError(f"time step t={t} is outside schedule for mode {mode_name!r}")

            traj_chunks = []
            xt_raw_chunks = []
            xrec_raw_chunks = []
            reverse_update_steps = list(range(t, -1, -1))

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                xb = x_raw_m11[start:end]

                forward = forward_to_xt(
                    schedule=bundle["schedule"],
                    precond=bundle["precond"],
                    x_raw_m11=xb,
                    t=t,
                    forward_seed=int(forward_seed_base + 100000 * t + start),
                )

                reverse = reconstruct_from_xt_with_trajectory(
                    model=bundle["model"],
                    schedule=bundle["schedule"],
                    xt=forward["xt_model"],
                    t_start=t,
                    reverse_seed=int(reverse_seed_base + 100000 * t + start),
                )

                traj_raw = _trajectory_model_to_raw(bundle["precond"], reverse["trajectory_model"]).detach().cpu()

                traj_chunks.append(traj_raw)
                xt_raw_chunks.append(traj_raw[:, 0].clone())
                xrec_raw_chunks.append(traj_raw[:, -1].clone())

            mode_store[t] = {
                "reverse_update_steps": reverse_update_steps,
                "xt_raw": torch.cat(xt_raw_chunks, dim=0),
                "x_rec_raw": torch.cat(xrec_raw_chunks, dim=0),
                "reverse_traj_raw": torch.cat(traj_chunks, dim=0),
            }

        bank["trajectories"][mode_name] = mode_store

    return bank