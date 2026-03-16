#!/usr/bin/env python3

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.vballnet_grid_v1a import VballNetGridV1a
from utils.grid_dataset import GridSequenceDataset


def parse_args():
    parser = argparse.ArgumentParser(description="GridTrackNet training")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--optimizer", choices=["Adadelta", "Adam", "AdamW", "SGD"], default="Adadelta")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["ReduceLROnPlateau", "None"], default="ReduceLROnPlateau")
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--seq", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--grid_rows", type=int, default=27)
    parser.add_argument("--grid_cols", type=int, default=48)
    parser.add_argument("--tol", type=int, default=4)
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--name", type=str, default="VballNetGridV1a")
    args = parser.parse_args()

    if args.lr is None:
        args.lr = {"Adadelta": 1.0, "Adam": 0.001, "AdamW": 0.001, "SGD": 0.01}[args.optimizer]
    return args


class GridTrackNetLoss(torch.nn.Module):
    def __init__(self, seq=5, conf_weight=1.0, offset_weight=0.001, alpha=0.75, gamma=2.0):
        super().__init__()
        self.seq = seq
        self.conf_weight = conf_weight
        self.offset_weight = offset_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-7

    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        y_pred = y_pred.view(batch_size, self.seq, 3, y_pred.shape[-2], y_pred.shape[-1]).permute(0, 1, 3, 4, 2)
        y_true = y_true.view(batch_size, self.seq, 3, y_true.shape[-2], y_true.shape[-1]).permute(0, 1, 3, 4, 2)

        conf_true = y_true[..., 0:1]
        x_true = y_true[..., 1:2]
        y_true_offset = y_true[..., 2:3]
        conf_pred = y_pred[..., 0:1].clamp(self.eps, 1.0 - self.eps)
        x_pred = y_pred[..., 1:2]
        y_pred_offset = y_pred[..., 2:3]

        offset_true = torch.cat([x_true, y_true_offset], dim=-1)
        offset_pred = torch.cat([x_pred, y_pred_offset], dim=-1)
        diff = torch.abs(offset_true - offset_pred).sum(dim=-1, keepdim=True)
        offset_loss = (conf_true * diff).sum(dim=(2, 3, 4)).mean(dim=1)

        positive = self.alpha * conf_true * torch.pow(1 - conf_pred, self.gamma) * torch.log(conf_pred)
        negative = (1 - self.alpha) * (1 - conf_true) * torch.pow(conf_pred, self.gamma) * torch.log(1 - conf_pred)
        confidence_loss = (-(positive + negative)).mean(dim=(1, 2, 3, 4))

        loss = self.offset_weight * offset_loss + self.conf_weight * confidence_loss
        return loss.mean()


def get_device(name):
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def create_optimizer(args, model):
    if args.optimizer == "Adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.optimizer == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)


def decode_prediction(pred, grid_cols, grid_rows, width, height):
    conf = pred[:, 0]
    x_offset = pred[:, 1]
    y_offset = pred[:, 2]
    conf_flat = conf.reshape(conf.shape[0], -1)
    max_idx = conf_flat.argmax(dim=1)
    rows = max_idx // grid_cols
    cols = max_idx % grid_cols

    batch_idx = torch.arange(conf.shape[0], device=pred.device)
    conf_score = conf[batch_idx, rows, cols]
    x = (cols.float() + x_offset[batch_idx, rows, cols]) * (width / grid_cols)
    y = (rows.float() + y_offset[batch_idx, rows, cols]) * (height / grid_rows)
    return conf_score, x, y


def compute_metrics(outputs, targets, seq, tol, grid_cols, grid_rows, width, height):
    outputs = outputs.view(outputs.shape[0], seq, 3, grid_rows, grid_cols)
    targets = targets.view(targets.shape[0], seq, 3, grid_rows, grid_cols)

    tp = tn = fp1 = fp2 = fn = 0
    for frame_idx in range(seq):
        pred_frame = outputs[:, frame_idx]
        true_frame = targets[:, frame_idx]
        pred_conf, pred_x, pred_y = decode_prediction(pred_frame, grid_cols, grid_rows, width, height)
        true_conf, true_x, true_y = decode_prediction(true_frame, grid_cols, grid_rows, width, height)

        pred_has_ball = pred_conf >= 0.5
        true_has_ball = true_conf >= 0.5

        for batch_idx in range(outputs.shape[0]):
            pred_visible = bool(pred_has_ball[batch_idx].item())
            true_visible = bool(true_has_ball[batch_idx].item())
            if not pred_visible and not true_visible:
                tn += 1
            elif pred_visible and not true_visible:
                fp2 += 1
            elif not pred_visible and true_visible:
                fn += 1
            else:
                distance = torch.sqrt((pred_x[batch_idx] - true_x[batch_idx]) ** 2 + (pred_y[batch_idx] - true_y[batch_idx]) ** 2)
                if float(distance.item()) > tol:
                    fp1 += 1
                else:
                    tp += 1

    accuracy = (tp + tn) / max(tp + tn + fp1 + fp2 + fn, 1)
    precision = tp / max(tp + fp1 + fp2, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = get_device(args.device)
        self.model = VballNetGridV1a(
            input_height=args.height,
            input_width=args.width,
            in_dim=args.seq * 3,
            out_dim=args.seq * 3,
        ).to(self.device)
        self.criterion = GridTrackNetLoss(seq=args.seq)
        self.optimizer = create_optimizer(args, self.model)
        self.scheduler = None
        if args.scheduler == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=args.factor,
                patience=args.patience,
                min_lr=args.min_lr,
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(args.out) / f"{args.name}_seq{args.seq}_{timestamp}"
        (self.save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        with (self.save_dir / "config.json").open("w") as f:
            json.dump(vars(args), f, indent=2)

    def setup_data(self):
        train_ds = GridSequenceDataset(
            self.args.data,
            seq=self.args.seq,
            stride=self.args.stride,
            height=self.args.height,
            width=self.args.width,
            grid_rows=self.args.grid_rows,
            grid_cols=self.args.grid_cols,
            augment=True,
        )
        val_ds = GridSequenceDataset(
            self.args.val_data,
            seq=self.args.seq,
            stride=self.args.stride,
            height=self.args.height,
            width=self.args.width,
            grid_rows=self.args.grid_rows,
            grid_cols=self.args.grid_cols,
            augment=False,
        )
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.args.batch,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=self.device.type == "cuda",
        )
        if len(train_ds) == 0:
            raise RuntimeError(f"Empty training dataset: {self.args.data}")
        if len(val_ds) == 0:
            raise RuntimeError(f"Empty validation dataset: {self.args.val_data}")
        print(f"Train samples: {len(train_ds)}")
        print(f"Val samples: {len(val_ds)}")

    def _run_epoch(self, loader, training):
        mode = "train" if training else "val"
        self.model.train(training)
        total_loss = 0.0
        metrics_sum = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        pbar = tqdm(loader, desc=mode, ncols=100)

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.set_grad_enabled(training):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()
            batch_metrics = compute_metrics(
                outputs.detach(),
                targets.detach(),
                seq=self.args.seq,
                tol=self.args.tol,
                grid_cols=self.args.grid_cols,
                grid_rows=self.args.grid_rows,
                width=self.args.width,
                height=self.args.height,
            )
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "f1": f"{batch_metrics['f1']:.4f}"})

        count = max(len(loader), 1)
        avg_metrics = {key: value / count for key, value in metrics_sum.items()}
        avg_loss = total_loss / count
        return avg_loss, avg_metrics

    def train(self):
        self.setup_data()
        best_val = float("inf")
        for epoch in range(self.args.epochs):
            start = time.time()
            train_loss, train_metrics = self._run_epoch(self.train_loader, training=True)
            val_loss, val_metrics = self._run_epoch(self.val_loader, training=False)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            latest_path = self.save_dir / "checkpoints" / "latest.pth"
            torch.save(checkpoint, latest_path)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(checkpoint, self.save_dir / "checkpoints" / "best.pth")

            elapsed = time.time() - start
            print(
                f"Epoch {epoch + 1}/{self.args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f} "
                f"time={elapsed:.1f}s"
            )

        print(f"Artifacts saved to {self.save_dir}")


if __name__ == "__main__":
    trainer = Trainer(parse_args())
    trainer.train()
