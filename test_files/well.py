# train_resnet_synthetic.py
# Minimal example: train ResNet on synthetic data
# Usage: python train_resnet_synthetic.py

import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Optional: use torchvision's resnet18 if available; otherwise fall back to a tiny ResNet-ish model
def build_model(num_classes: int, in_channels: int = 3):
    try:
        import torchvision.models as models
        model = models.resnet18(weights=None)
        # Adjust first conv if channels != 3
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Adjust classifier head
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    except Exception:
        # Tiny fallback if torchvision isn't installed
        def conv_bn_relu(cin, cout, k, s, p):
            return nn.Sequential(
                nn.Conv2d(cin, cout, k, s, p, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        class TinyResNet(nn.Module):
            def __init__(self, in_ch, num_cls):
                super().__init__()
                self.stem = conv_bn_relu(in_ch, 32, 3, 1, 1)
                self.layer1 = nn.Sequential(conv_bn_relu(32, 32, 3, 1, 1),
                                            conv_bn_relu(32, 32, 3, 1, 1))
                self.layer2 = nn.Sequential(conv_bn_relu(32, 64, 3, 2, 1),
                                            conv_bn_relu(64, 64, 3, 1, 1))
                self.layer3 = nn.Sequential(conv_bn_relu(64, 128, 3, 2, 1),
                                            conv_bn_relu(128, 128, 3, 1, 1))
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(128, num_cls)

            def forward(self, x):
                x = self.stem(x)
                x = self.layer1(x) + x  # simple residual-ish skip (same shape)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.pool(x).flatten(1)
                return self.fc(x)

        return TinyResNet(in_channels, num_classes)

class SyntheticImageDataset(Dataset):
    """
    Generates random images and labels on-the-fly.
    Optionally injects simple patterns to make the task non-trivial.
    """
    def __init__(self, length=10_000, image_size=(3, 224, 224), num_classes=10, seed=42, patterned=True):
        super().__init__()
        self.length = length
        self.C, self.H, self.W = image_size
        self.num_classes = num_classes
        self.patterned = patterned
        g = torch.Generator().manual_seed(seed)
        # Pre-generate labels for determinism and class balance
        reps = math.ceil(length / num_classes)
        labels = torch.arange(num_classes).repeat(reps)[:length]
        self.labels = labels[torch.randperm(length, generator=g)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        y = int(self.labels[idx])
        # Base noise
        x = torch.randn(self.C, self.H, self.W, dtype=torch.float32) * 0.5
        if self.patterned:
            # Add a class-dependent blob so the model can learn something
            s = max(4, min(self.H, self.W) // 8)
            cy = (y * 37) % (self.H - s)  # pseudo-random but deterministic by class
            cx = (y * 53) % (self.W - s)
            x[:, cy:cy + s, cx:cx + s] += 2.0  # brighter square
        return x, y

def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean()

def main():
    # -------------------- Config --------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 10
    in_channels = 3
    image_size = (in_channels, 224, 224)  # (C, H, W). Use (3, 32, 32) for CIFAR-like
    train_len = 8_000
    val_len = 2_000
    batch_size = 64
    epochs = 5
    lr = 5e-4
    weight_decay = 1e-2
    use_amp = torch.cuda.is_available()
    workers = 4

    # -------------------- Data ----------------------
    train_ds = SyntheticImageDataset(length=train_len, image_size=image_size, num_classes=num_classes, patterned=True)
    val_ds = SyntheticImageDataset(length=val_len, image_size=image_size, num_classes=num_classes, patterned=True, seed=123)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # -------------------- Model/Opt -----------------
    model = build_model(num_classes=num_classes, in_channels=in_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -------------------- Train/Val -----------------
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_acc += accuracy(logits.detach(), yb).item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        train_acc = running_acc / max(1, n_batches)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_batches = 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                val_acc += accuracy(logits, yb).item()
                n_batches += 1
        val_loss /= max(1, n_batches)
        val_acc /= max(1, n_batches)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | "
              f"{dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(),
                        "num_classes": num_classes,
                        "in_channels": in_channels}, "best_resnet_synth.pt")

    print(f"Done. Best val acc: {best_val_acc:.3f}. Weights saved to best_resnet_synth.pt")

if __name__ == "__main__":
    main()
