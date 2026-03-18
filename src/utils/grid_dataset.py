import csv
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class GridSequenceDataset(Dataset):
    def __init__(
        self,
        root_dir,
        seq=5,
        stride=2,
        height=432,
        width=768,
        grid_rows=27,
        grid_cols=48,
        augment=False,
        grayscale=False,
    ):
        self.root_dir = Path(root_dir)
        self.seq = seq
        self.stride = stride
        self.height = height
        self.width = width
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.augment = augment
        self.grayscale = grayscale
        self.grid_size_col = width / grid_cols
        self.grid_size_row = height / grid_rows
        self.samples = self._scan_dataset()

    def _scan_dataset(self):
        items = []
        match_dirs = sorted(
            match_dir
            for match_dir in self.root_dir.iterdir()
            if match_dir.is_dir() and match_dir.name.startswith("match")
        )
        for match_dir in match_dirs:
            inputs_dir = match_dir / "inputs"
            annotations_dir = match_dir / "annotations"
            if not inputs_dir.exists() or not annotations_dir.exists():
                continue
            for annotation_path in sorted(annotations_dir.glob("*.csv")):
                sequence_name = annotation_path.stem
                frames_dir = inputs_dir / sequence_name
                if not frames_dir.exists():
                    continue
                rows = self._read_rows(annotation_path)
                frame_paths = sorted(frames_dir.glob("*.png"), key=lambda p: int(p.stem))
                if len(rows) != len(frame_paths) or len(rows) < self.seq:
                    continue
                for start_idx in range(0, len(rows) - self.seq + 1, self.stride):
                    items.append(
                        {
                            "frame_paths": frame_paths[start_idx : start_idx + self.seq],
                            "rows": rows[start_idx : start_idx + self.seq],
                            "match": match_dir.name,
                            "sequence": sequence_name,
                            "start_idx": start_idx,
                        }
                    )
        return items

    def _read_rows(self, annotation_path):
        rows = []
        with annotation_path.open(newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append(
                    {
                        "frame": int(row["Frame"]),
                        "visibility": int(row["Visibility"]),
                        "x": float(row["X"]) if row["X"] != "" else 0.0,
                        "y": float(row["Y"]) if row["Y"] != "" else 0.0,
                    }
                )
        return rows

    def _make_target(self, row, flipped=False):
        conf = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        x_offset = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        y_offset = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        visible = not (
            row["visibility"] == 0 and abs(row["x"]) < 1e-6 and abs(row["y"]) < 1e-6
        )
        if visible:
            x = row["x"]
            y = row["y"]
            if flipped:
                x = (self.width - 1) - x

            x = float(np.clip(x, 0, self.width - 1))
            y = float(np.clip(y, 0, self.height - 1))
            x_pos = x / self.grid_size_col
            y_pos = y / self.grid_size_row
            x_cell = min(int(x_pos), self.grid_cols - 1)
            y_cell = min(int(y_pos), self.grid_rows - 1)
            conf[y_cell, x_cell] = 1.0
            x_offset[y_cell, x_cell] = x_pos - x_cell
            y_offset[y_cell, x_cell] = y_pos - y_cell

        return np.stack((conf, x_offset, y_offset), axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        flipped = self.augment and random.random() < 0.5
        frames = []
        targets = []

        for frame_path, row in zip(sample["frame_paths"], sample["rows"]):
            read_mode = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
            image = cv2.imread(str(frame_path), read_mode)
            if image is None:
                if self.grayscale:
                    image = np.zeros((self.height, self.width), dtype=np.uint8)
                else:
                    image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                image = cv2.resize(image, (self.width, self.height))
            if self.grayscale:
                if flipped:
                    image = np.ascontiguousarray(np.flip(image, axis=1))
                image = image.astype(np.float32) / 255.0
                image = image[None, ...]
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if flipped:
                    image = np.ascontiguousarray(np.flip(image, axis=1))
                image = image.astype(np.float32) / 255.0
                image = np.transpose(image, (2, 0, 1))
            frames.append(image)
            targets.append(self._make_target(row, flipped=flipped))

        x = torch.from_numpy(np.concatenate(frames, axis=0))
        y = torch.from_numpy(np.concatenate(targets, axis=0))
        return x, y
