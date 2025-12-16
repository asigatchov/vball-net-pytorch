"""
TrackNet Training Script

Usage Examples:
python train.py --data dataset/train
python train.py --data dataset/train --batch 8 --epochs 50 --lr 0.001
python train.py --data dataset/train --optimizer Adam --lr 0.001 --batch 16 --plot 10
python train.py --resume best.pth --data dataset/train --lr 0.0001
python train.py --resume checkpoint.pth --data dataset/train --optimizer Adam --epochs 100
python train.py --data training_data/train --batch 3 --lr 1  --optimizer Adadelta


Parameters:
--data: Training dataset path (required)
--resume: Checkpoint path for resuming
--split: Train/val split ratio (default: 0.8)
--seed: Random seed (default: 26)
--batch: Batch size (default: 3)
--epochs: Training epochs (default: 30)
--workers: Data loader workers (default: 0)
--device: Device auto/cpu/cuda/mps (default: auto)
--optimizer: Adadelta/Adam/AdamW/SGD (default: Adadelta)
--lr: Learning rate (default: auto per optimizer)
--wd: Weight decay (default: 0)
--scheduler: ReduceLROnPlateau/None (default: ReduceLROnPlateau)
--factor: LR reduction factor (default: 0.5)
--patience: LR reduction patience (default: 3)
--min_lr: Minimum learning rate (default: 1e-6)
--plot: Loss plot interval (default: 1)
--out: Output directory (default: outputs)
--name: Experiment name (default: exp)
"""

import argparse
import json
import signal
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model.loss import WeightedBinaryCrossEntropy
from utils.tracknet_datasetv2 import FrameHeatmapDataset
import os
import numpy as np
import cv2
import torch  # Add torch import for tensor operations
from torch.utils.tensorboard import SummaryWriter


# Choose the version of TrackNet model you want to use
# from model.vballnet_v1 import VballNetV1 as VballNetV1a
from model.vballnet_v1a import VballNetV1a as VballNetV1a

from model.vballnet_v2 import VballNetV2
from model.vballnet_v3 import VballNetV3
from model.vballnet_v3b import VballNetV3b
from model.vballnet_v3c import VballNetV3c
from model.vballnet_v1c import VballNetV1c

# Available models
AVAILABLE_MODELS = ["TrackNet", "VballNetV2", "VballNetV3b", "VballNetV3c", "VballNetV3", "VballNetV1a", "VballNetV1c", "VballNetV1d", "VballNetFastV1", "VballNetFastV2"]


class Trainer:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.interrupted = False
        self.best_loss = float("inf")
        self.device = self._get_device()
        self._setup_dirs()
        self._load_checkpoint()
        self.losses = {"batch": [], "steps": [], "lrs": [], "train": [], "val": []}
        self.step = 0
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.save_dir / "tensorboard")
        signal.signal(signal.SIGINT, self._interrupt)
        signal.signal(signal.SIGTERM, self._interrupt)

    def _get_device(self):
        if self.args.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(self.args.device)

    def _setup_dirs(self):
        print("Setting up output directories...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_resumed" if self.args.resume else ""
        # Формируем составное имя модели для папки и чекпоинтов
        model_tag = f"{self.args.model_name}_seq{self.args.seq}" + ("_grayscale" if self.args.grayscale else "")
        self.save_dir = Path(self.args.out) / f"{self.args.name}_{model_tag}{suffix}_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=2)
        print(f"Output directory created: {self.save_dir}")

    def _load_checkpoint(self):
        if not self.args.resume:
            return
        print("Loading checkpoint...")
        path = Path(self.args.resume)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.checkpoint = torch.load(path, map_location="cpu")
        self.start_epoch = self.checkpoint["epoch"] + (
            0 if self.checkpoint.get("is_emergency", False) else 1
        )
        print(
            f"Checkpoint loaded, resuming from epoch \033[93m{self.start_epoch + 1}\033[0m"
        )

    def _interrupt(self, signum, frame):
        print("\n\033[91mInterrupt detected\033[0m, saving emergency checkpoint...")
        self.interrupted = True

    def _calculate_effective_lr(self):
        if self.args.optimizer == "Adadelta":
            if not hasattr(self.optimizer, "state") or not self.optimizer.state:
                return self.args.lr

            effective_lrs = []
            eps = self.optimizer.param_groups[0].get("eps", 1e-6)

            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.optimizer.state[p]
                    if len(state) == 0:
                        continue

                    square_avg = state.get("square_avg")
                    acc_delta = state.get("acc_delta")

                    if square_avg is not None and acc_delta is not None:
                        if torch.is_tensor(square_avg) and torch.is_tensor(acc_delta):
                            rms_delta = (acc_delta + eps).sqrt().mean()
                            rms_grad = (square_avg + eps).sqrt().mean()
                            if rms_grad > eps:
                                effective_lr = self.args.lr * rms_delta / rms_grad
                                effective_lrs.append(effective_lr.item())

            if effective_lrs:
                avg_lr = sum(effective_lrs) / len(effective_lrs)
                return max(avg_lr, eps)
            else:
                return self.args.lr
        else:
            return self.optimizer.param_groups[0]["lr"]

    def setup_data(self):
        print("Loading dataset...")
        dataset = FrameHeatmapDataset(
            self.args.data,
            seq=self.args.seq,
            grayscale=self.args.grayscale
        )
        print(f"Dataset loaded: \033[94m{len(dataset)}\033[0m samples {self.args.grayscale} sequences  {self.args.seq}")

        if self.args.val_data:
            print("Loading separate validation dataset...")
            val_dataset = FrameHeatmapDataset(
                self.args.val_data,
                seq=self.args.seq,
                grayscale=self.args.grayscale
            )
            train_ds = dataset
            val_ds = val_dataset
        else:
            print("Splitting dataset...")
            torch.manual_seed(self.args.seed)
            train_size = int(self.args.split * len(dataset))
            train_ds, val_ds = random_split(
                dataset, [train_size, len(dataset) - train_size]
            )

        print("Creating data loaders...")
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
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=self.device.type == "cuda",
        )
        print(
            f"Data loaders ready - Train: \033[94m{len(train_ds)}\033[0m | Val: \033[94m{len(val_ds)}\033[0m"
        )

    def _create_optimizer(self):
        optimizers = {
            "Adadelta": lambda: torch.optim.Adadelta(
                self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd
            ),
            "Adam": lambda: torch.optim.Adam(
                self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd
            ),
            "AdamW": lambda: torch.optim.AdamW(
                self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd
            ),
            "SGD": lambda: torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.wd,
                momentum=0.9,
            ),
        }
        return optimizers[self.args.optimizer]()

    def setup_model(self):
        print("Initializing model...")
        if self.args.grayscale:
            in_dim = self.args.seq
            out_dim = self.args.seq
        else:
            in_dim = self.args.seq * 3
            out_dim = self.args.seq
        
        # Define common parameters to reduce duplication
        common_params = {
            "height": 288,
            "width": 512,
            "in_dim": in_dim,
            "out_dim": out_dim
        }

        if self.args.model_name == "VballNetV1a" or self.args.model_name == "TrackNet":
            self.model = VballNetV1a(**common_params).to(self.device)
            # Use setattr to avoid type checking issues
            setattr(self.model, '_model_type', "VballNetV1a")

        elif self.args.model_name == "VballNetV1c":
            self.model = VballNetV1c(**common_params).to(self.device)
            setattr(self.model, '_model_type', "VballNetV1c")

        elif self.args.model_name == "VballNetV2":
            self.model = VballNetV2(**common_params).to(self.device)
            setattr(self.model, '_model_type', "VballNetV2")

        elif self.args.model_name == "VballNetV3b":
            self.model = VballNetV3b(**common_params).to(self.device)
            setattr(self.model, '_model_type', "VballNetV3b")

        elif self.args.model_name == "VballNetV3c":
            self.model = VballNetV3c(**common_params).to(self.device)
            setattr(self.model, '_model_type', "VballNetV3c")

        elif self.args.model_name == "VballNetV3":
            self.model = VballNetV3(**common_params).to(self.device)
            setattr(self.model, '_model_type', "VballNetV3")

        elif 'VballNetFastV1' in self.args.model_name:
            # Using standardized parameters for fast models
            fast_params = {
                "input_height": 288,
                "input_width": 512,
                "in_dim": in_dim,
                "out_dim": out_dim,
                "channels": [8, 16, 32],
                "bottleneck_channels": 64,
                "dropout_p": 0.2
            }
            self.model = VballNetFastV1(**fast_params).to(self.device)
            setattr(self.model, '_model_type', "VballNetFastV1")

        elif 'VballNetFastV2' in self.args.model_name:
            # Using standardized parameters for fast models
            fast_params = {
                "input_height": 288,
                "input_width": 512,
                "in_dim": in_dim,
                "out_dim": out_dim,
                "channels": [8, 16, 32],
                "bottleneck_channels": 64,
                "dropout_p": 0.2
            }
            self.model = VballNetFastV2(**fast_params).to(self.device)
            setattr(self.model, '_model_type', "VballNetFastV2")

        else:
            raise ValueError(f"Unknown model: {self.args.model_name}")

        self.criterion = WeightedBinaryCrossEntropy()
        self.optimizer = self._create_optimizer()

        if self.args.scheduler == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.args.factor,
                patience=self.args.patience,
                min_lr=self.args.min_lr,
            )
        else:
            self.scheduler = None

        if hasattr(self, "checkpoint"):
            print("Loading model state from checkpoint...")
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
            print("Model state loaded successfully")

        print(
            f"Model ready - Optimizer: \033[93m{self.args.optimizer}\033[0m | LR: \033[93m{self.args.lr}\033[0m | WD: \033[93m{self.args.wd}\033[0m"
        )

    def save_checkpoint(self, epoch, train_loss, val_loss, is_emergency=False):
        print("Saving checkpoint...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Формируем составное имя модели для чекпоинта
        model_tag = f"{self.args.model_name}_seq{self.args.seq}" + ("_grayscale" if self.args.grayscale else "")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "is_emergency": is_emergency,
            "history": self.losses.copy(),
            "step": self.step,
            "timestamp": timestamp,
        }

        prefix = "emergency_" if is_emergency else "checkpoint_"
        filename = f"{prefix}{model_tag}_epoch_{epoch + 1}_{timestamp}.pth"
        filepath = self.save_dir / "checkpoints" / filename
        torch.save(checkpoint, filepath)

        if not is_emergency and val_loss < self.best_loss:
            self.best_loss = val_loss
            best_name = f"{model_tag}_best.pth"
            torch.save(checkpoint, self.save_dir / "checkpoints" / best_name)
            print(f"Checkpoint saved: {filename} (\033[92mBest model updated\033[0m)")
            return filepath, True

        print(f"Checkpoint saved: {filename}")
        return filepath, False


    def plot_curves(self, epoch):
        print("Generating training plots...")

        # Create a single figure with three subplots in a row
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

        # Plot 1: Train and Val Loss
        if self.losses["train"]:
            cnt = len(self.losses["train"])
            start = 1
            epochs = list(range(start, cnt + 1))
            ax1.plot(epochs, self.losses["train"], "bo-", label="Train Loss")
            ax1.plot(epochs, self.losses["val"], "ro-", label="Val Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Train and Validation Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Batch Loss
        if self.losses["batch"]:
            ax2.plot(
                self.losses["steps"],
                self.losses["batch"],
                "b-",
                alpha=0.3,
                label="Batch Loss",
            )
            ax2.set_xlabel("Batch")
            ax2.set_ylabel("Loss")
            ax2.set_title("Batch Loss")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Plot 3: Learning Rate
        if self.losses["lrs"]:
            ax3.plot(self.losses["steps"], self.losses["lrs"], "g-")
            ax3.set_xlabel("Batch")
            ax3.set_ylabel("Learning Rate")
            ax3.set_title("Learning Rate")
            ax3.set_yscale("log")
            ax3.grid(True, alpha=0.3)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the combined plot
        plt.savefig(
            self.save_dir / "plots" / f"training_metrics_epoch_{epoch + 1}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Training plots saved for epoch \033[93m{epoch + 1}\033[0m")

    def validate(self):
        print("Starting validation...")
        self.model.eval()
        total_loss = 0.0
        vis_dir = self.save_dir / "val_vis"

        # Metrics tracking
        total_f1 = 0.0
        total_accuracy = 0.0
        sample_count = 0

        vis_dir.mkdir(exist_ok=True)
        max_vis_batches = 5  # Сколько батчей визуализировать
        use_gru = hasattr(self.model, '_model_type') and self.model._model_type == "VballNetV1c"
        h0 = None  # Начальное состояние GRU
        with torch.no_grad():
            val_pbar = tqdm(total=len(self.val_loader), desc="Validation", ncols=100)
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                if self.interrupted:
                    val_pbar.close()
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if use_gru:
                    try:
                        outputs, hn = self.model(inputs, h0=h0)
                        h0 = hn.detach()
                    except Exception as e:
                        outputs, hn = self.model(inputs, h0=None)
                        h0 = hn.detach()
                else:
                    outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Calculate metrics for the central frame
                batch_f1, batch_accuracy = self._calculate_metrics(outputs, targets)
                total_f1 += batch_f1
                total_accuracy += batch_accuracy
                sample_count += inputs.size(0)

                # --- Визуализация ---
                if batch_idx < max_vis_batches:
                    # inputs: (B, C, H, W), outputs: (B, seq, H, W), targets: (B, seq, H, W)
                    inp = inputs[0].detach().cpu()  # (C, H, W)
                    pred = outputs[0].detach().cpu()  # (seq, H, W)
                    gt = targets[0].detach().cpu()   # (seq, H, W)
                    # Определяем число кадров для визуализации
                    n_vis = min(pred.shape[0], gt.shape[0], 9)
                    for i in range(n_vis):
                        # Входной кадр (если grayscale - берем 1 канал, если RGB - 3)
                        if inp.shape[0] == pred.shape[0]:
                            # grayscale
                            rgb = np.stack([inp[i].numpy()]*3, axis=2)
                        else:
                            rgb = inp[i*3:(i+1)*3].permute(1, 2, 0).numpy()
                        rgb = (rgb * 255).astype(np.uint8)
                        # Предсказанный heatmap
                        pred_hm = pred[i].numpy()
                        pred_hm = (pred_hm * 255).astype(np.uint8)
                        pred_hm_color = cv2.applyColorMap(pred_hm, cv2.COLORMAP_JET)
                        # Эталонный heatmap
                        gt_hm = gt[i].numpy()
                        gt_hm = (gt_hm * 255).astype(np.uint8)
                        gt_hm_color = cv2.applyColorMap(gt_hm, cv2.COLORMAP_JET)
                        # Overlay
                        overlay_pred = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.6, pred_hm_color, 0.4, 0)
                        overlay_gt = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.6, gt_hm_color, 0.4, 0)
                        # Собираем в одну картинку
                        vis_img = np.vstack([
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            overlay_pred,
                            overlay_gt
                        ])
                        vis_path = vis_dir / f"val_batch{batch_idx}_frame{i}.jpg"
                        cv2.imwrite(str(vis_path), vis_img)
                # --- конец визуализации ---

                val_pbar.update(1)
                val_pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            val_pbar.close()

        avg_loss = total_loss / len(self.val_loader)
        avg_f1 = total_f1 / len(self.val_loader)
        avg_accuracy = total_accuracy / len(self.val_loader)
        
        print(f"Validation completed - Average loss: \033[94m{avg_loss:.6f}\033[0m")
        print(f"Validation F1 Score: \033[94m{avg_f1:.6f}\033[0m")
        print(f"Validation Accuracy (dist ≤ 10px): \033[94m{avg_accuracy:.6f}\033[0m")
        return avg_loss, avg_f1, avg_accuracy

    def _calculate_metrics(self, predictions, targets):
        """
        Calculate F1 score and accuracy for the central frame.
        
        Args:
            predictions: Model outputs tensor of shape (B, seq, H, W)
            targets: Ground truth tensor of shape (B, seq, H, W)
            
        Returns:
            f1_score: F1 score at threshold 0.5
            accuracy: Accuracy at distance <= 10 pixels
        """
        # Focus on the central frame
        batch_size = predictions.size(0)
        seq_len = predictions.size(1)
        center_idx = seq_len // 2
        
        # Get predictions and targets for the central frame
        pred_center = predictions[:, center_idx, :, :]  # (B, H, W)
        target_center = targets[:, center_idx, :, :]    # (B, H, W)
        
        # Flatten for easier computation
        pred_flat = pred_center.view(batch_size, -1)    # (B, H*W)
        target_flat = target_center.view(batch_size, -1)  # (B, H*W)
        
        # F1 Score calculation at threshold 0.5
        pred_binary = (pred_flat > 0.5).float()
        target_binary = (target_flat > 0.5).float()
        
        # Calculate TP, FP, FN
        tp = (pred_binary * target_binary).sum(dim=1)
        fp = (pred_binary * (1 - target_binary)).sum(dim=1)
        fn = ((1 - pred_binary) * target_binary).sum(dim=1)
        
        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        f1_score = f1_score.mean().item()
        
        # Accuracy calculation (distance <= 10 pixels)
        # Find predicted and ground truth positions
        pred_positions = self._find_positions(pred_flat)
        target_positions = self._find_positions(target_flat)
        
        # Calculate distances
        distances = torch.sqrt(((pred_positions - target_positions) ** 2).sum(dim=1))
        accuracy = (distances <= 10).float().mean().item()
        
        return f1_score, accuracy
    
    def _find_positions(self, heatmaps):
        """
        Find the position of maximum activation in each heatmap.
        
        Args:
            heatmaps: Tensor of shape (B, H*W)
            
        Returns:
            positions: Tensor of shape (B, 2) with (x, y) coordinates
        """
        batch_size, flat_dim = heatmaps.shape
        H, W = 288, 512  # Fixed dimensions from dataset
        
        # Get indices of maximum values
        max_indices = torch.argmax(heatmaps, dim=1)  # (B,)
        
        # Convert flat indices to (y, x) coordinates
        y_coords = (max_indices // W).float()
        x_coords = (max_indices % W).float()
        
        # Return as (x, y) coordinates
        return torch.stack([x_coords, y_coords], dim=1)  # (B, 2)

    def train(self):
        print(f"Starting training on \033[93m{self.device}\033[0m")
        self.setup_data()
        self.setup_model()
        use_gru = hasattr(self.model, '_model_type') and self.model._model_type == "VballNetV1c"

        for epoch in range(self.start_epoch, self.args.epochs):
            if self.interrupted:
                break

            print(
                f"\nEpoch \033[95m{epoch + 1}\033[0m/\033[95m{self.args.epochs}\033[0m"
            )
            start_time = time.time()
            self.model.train()
            total_loss = 0.0
            h0 = None  # Начальное состояние GRU
            train_pbar = tqdm(total=len(self.train_loader), desc=f"Training", ncols=100)
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                if self.interrupted:
                    train_pbar.close()
                    print("Emergency save triggered...")
                    val_loss, val_f1, val_accuracy = self.validate()
                    self.save_checkpoint(
                        epoch, total_loss / (batch_idx + 1), val_loss, True
                    )
                    self.plot_curves(epoch)
                    return

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # --- MIXUP augmentation ---
                if self.args.alpha is not None and self.args.alpha > 0:
                    inputs, targets = mixup(inputs, targets, self.args.alpha)
                # --- END MIXUP ---

                self.optimizer.zero_grad()
                if use_gru:
                    try:
                        outputs, hn = self.model(inputs, h0=h0)
                        h0 = hn.detach()
                    except Exception as e:
                        outputs, hn = self.model(inputs, h0=None)
                        h0 = hn.detach()
                else:
                    outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss
                self.step += 1

                current_lr = self._calculate_effective_lr()

                if self.step % self.args.plot == 0:
                    self.losses["batch"].append(batch_loss)
                    self.losses["steps"].append(self.step)
                    self.losses["lrs"].append(current_lr)

                train_pbar.update(1)
                train_pbar.set_postfix(
                    {"loss": f"{batch_loss:.6f}", "lr": f"{current_lr:.2e}"}
                )
            train_pbar.close()

            train_loss = total_loss / len(self.train_loader)
            val_loss, val_f1, val_accuracy = self.validate()

            self.losses["train"].append(train_loss)
            self.losses["val"].append(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time

            print(
                f"Epoch [\033[95m{epoch + 1}\033[0m/\033[95m{self.args.epochs}\033[0m] Train: \033[94m{train_loss:.6f}\033[0m Val: \033[94m{val_loss:.6f}\033[0m "
                f"LR: \033[94m{current_lr:.6e}\033[0m Time: \033[94m{elapsed:.1f}s\033[0m"
            )

            # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Metrics/F1_Score', val_f1, epoch)
            self.writer.add_scalar('Metrics/Accuracy_10px', val_accuracy, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            if self.scheduler:
                print("Updating learning rate scheduler...")
                self.scheduler.step(val_loss)

            _, is_best = self.save_checkpoint(epoch, train_loss, val_loss)
            if is_best:
                print(f"\033[92mNew best model! Val Loss: {val_loss:.6f}\033[0m")

            self.plot_curves(epoch)

        if not self.interrupted:
            print("\n\033[92mTraining completed successfully!\033[0m")
            print(f"\033[92mAll results saved to: {self.save_dir}\033[0m")
            # Close TensorBoard writer
            self.writer.close()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
