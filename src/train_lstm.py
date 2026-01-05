import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fit(model, train_ds, val_ds, cfg, ckpt_path):
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best = float("inf")
    bad = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_losses.append(loss.item())

        tr = float(np.mean(tr_losses))
        va = float(np.mean(va_losses))
        print(f"[Epoch {ep:02d}] train_loss={tr:.6f} val_loss={va:.6f}")

        if va < best:
            best = va
            bad = 0
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Early stopping. Best val_loss={best:.6f}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    return model
