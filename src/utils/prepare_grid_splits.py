#!/usr/bin/env python3

import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build balanced train/val/test splits for grid data")
    parser.add_argument("--source", type=str, required=True, help="Raw dataset root with catalog folders")
    parser.add_argument("--output", type=str, required=True, help="Split output root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def select_counts(total, train_ratio, val_ratio, test_ratio):
    if total < 3:
        raise ValueError(f"Need at least 3 videos per catalog to build train/val/test splits, got {total}")

    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    counts = {
        split: max(1, int(round(total * ratio)))
        for split, ratio in ratios.items()
    }

    while sum(counts.values()) > total:
        for split in ("train", "val", "test"):
            if sum(counts.values()) <= total:
                break
            if counts[split] > 1:
                counts[split] -= 1

    while sum(counts.values()) < total:
        counts["train"] += 1

    return counts


def safe_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main():
    args = parse_args()
    source_root = Path(args.source)
    output_root = Path(args.output)

    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    if output_root.exists():
        if args.force:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --force to replace it.")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "splits": {}}
    split_roots = {name: output_root / name for name in ("train", "val", "test")}
    for split_root in split_roots.values():
        split_root.mkdir(parents=True, exist_ok=True)

    catalogs = sorted([path for path in source_root.iterdir() if path.is_dir() and not path.name.startswith(".")])
    rng = random.Random(args.seed)

    for catalog_idx, catalog in enumerate(catalogs):
        video_dir = catalog / "video"
        csv_dir = catalog / "csv"
        if not video_dir.exists() or not csv_dir.exists():
            continue

        video_paths = sorted(video_dir.glob("*.mp4"))
        csv_paths = sorted(csv_dir.glob("*_ball.csv"))
        csv_by_stem = {path.stem.replace("_ball", ""): path for path in csv_paths}
        stems = [path.stem for path in video_paths if path.stem in csv_by_stem]
        if len(stems) < 3:
            raise ValueError(f"Catalog {catalog.name} has only {len(stems)} paired videos; need at least 3.")

        catalog_rng = random.Random(args.seed + catalog_idx)
        catalog_rng.shuffle(stems)
        counts = select_counts(len(stems), args.train_ratio, args.val_ratio, args.test_ratio)

        catalog_manifest = {
            "total": len(stems),
            "train": stems[: counts["train"]],
            "val": stems[counts["train"] : counts["train"] + counts["val"]],
            "test": stems[counts["train"] + counts["val"] :],
        }
        manifest["splits"][catalog.name] = {
            "train": catalog_manifest["train"],
            "val": catalog_manifest["val"],
            "test": catalog_manifest["test"],
        }

        for split_name, split_stems in catalog_manifest.items():
            if split_name == "total":
                continue
            split_catalog_dir = split_roots[split_name] / catalog.name
            (split_catalog_dir / "video").mkdir(parents=True, exist_ok=True)
            (split_catalog_dir / "csv").mkdir(parents=True, exist_ok=True)
            for stem in split_stems:
                safe_symlink(video_dir / f"{stem}.mp4", split_catalog_dir / "video" / f"{stem}.mp4")
                safe_symlink(csv_by_stem[stem], split_catalog_dir / "csv" / f"{stem}_ball.csv")

        print(
            f"{catalog.name}: "
            f"train={len(catalog_manifest['train'])} "
            f"val={len(catalog_manifest['val'])} "
            f"test={len(catalog_manifest['test'])}"
        )

    with (output_root / "split_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Split root prepared at {output_root}")


if __name__ == "__main__":
    main()
