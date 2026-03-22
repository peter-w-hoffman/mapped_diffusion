import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import device

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_classifier_transform(input_space="01_norm"):
    """
    Keep the original notebook behavior by default:
      "01_norm" = ToTensor() + MNIST normalization
    """
    if input_space == "01_norm":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])
    # if input_space == "01_raw":
    #     return transforms.ToTensor()
    raise ValueError(f"Unknown INPUT_SPACE: {input_space}")


def ddpm_raw_to_clf_input(x_raw_m11, input_space="01_norm", mnist_mean=MNIST_MEAN, mnist_std=MNIST_STD):
    """
    Convert DDPM raw output in [-1,1] into the classifier input space.
    """
    x_raw_m11 = x_raw_m11.clamp(-1, 1)
    x01 = (x_raw_m11 + 1.0) / 2.0

    if input_space == "01_norm":
        return (x01 - mnist_mean) / mnist_std
    # if input_space == "01_raw":
    #     return x01
    raise ValueError(f"Unknown INPUT_SPACE: {input_space}")


def get_classifier_loaders(root="./data", batch_size=64, input_space="01_norm"):
    tfm = get_classifier_transform(input_space=input_space)

    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return train_ds, test_ds, train_loader, test_loader


class MNISTCNN(nn.Module):
    """
    Same classifier architecture as your notebook.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = self.drop(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return 100.0 * correct / max(total, 1)


def train_classifier(model, train_loader, test_loader, epochs=5, lr=1e-3, weight_decay=1e-4, device=device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[CLF] Epoch {epoch}/{epochs}", leave=True)
        ema = None

        for x, y in pbar:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            ema = loss.item() if ema is None else (0.95 * ema + 0.05 * loss.item())
            pbar.set_postfix(loss=f"{ema:.4f}")

        acc = eval_acc(model, test_loader)
        print(f"[CLF] Epoch {epoch}: loss={ema:.4f} | test_acc={acc:.2f}%")

    return model


@torch.no_grad()
def classify_raw_m11(model, x_raw_m11, input_space="01_norm", mnist_mean=MNIST_MEAN, mnist_std=MNIST_STD):
    """
    Classifier evaluation entry point for reconstructed DDPM outputs.
    """
    model.eval()
    x_clf = ddpm_raw_to_clf_input(
        x_raw_m11,
        input_space=input_space,
        mnist_mean=mnist_mean,
        mnist_std=mnist_std,
    ).to(next(model.parameters()).device)

    logits = model(x_clf)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    confidences = probs.max(dim=1).values

    return {
        "logits": logits,
        "probs": probs,
        "preds": preds,
        "confidences": confidences,
        "x_clf": x_clf,
    }


@torch.no_grad()
def classify_and_show_raw(model, x_raw_m11, input_space="01_norm", mnist_mean=MNIST_MEAN, mnist_std=MNIST_STD, nrow=5):
    """
    Convenience function preserving the original DDPM-samples-to-classifier visualization workflow.
    """
    out = classify_raw_m11(
        model,
        x_raw_m11,
        input_space=input_space,
        mnist_mean=mnist_mean,
        mnist_std=mnist_std,
    )

    vis = (x_raw_m11.clamp(-1, 1) + 1.0) / 2.0
    grid = make_grid(vis, nrow=nrow, padding=6)

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).squeeze().cpu(), cmap="gray")
    plt.show()

    print("preds:", out["preds"].detach().cpu().tolist())
    print("confidences:", [round(x, 4) for x in out["confidences"].detach().cpu().tolist()])
    return out


def save_classifier_checkpoint(path, model, input_space="01_norm", mnist_mean=MNIST_MEAN, mnist_std=MNIST_STD):
    ckpt = {
        "model_state": model.state_dict(),
        "input_space": input_space,
        "mnist_mean": mnist_mean,
        "mnist_std": mnist_std,
    }
    torch.save(ckpt, path)


def load_classifier_checkpoint(path, map_location="cpu", device=device):
    ckpt = torch.load(path, map_location=map_location)
    model = MNISTCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cfg = {
        "input_space": ckpt.get("input_space", "01_norm"),
        "mnist_mean": ckpt.get("mnist_mean", MNIST_MEAN),
        "mnist_std": ckpt.get("mnist_std", MNIST_STD),
    }
    return model, cfg, ckpt