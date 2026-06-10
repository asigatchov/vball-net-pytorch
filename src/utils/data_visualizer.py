"""
Dataset Training Player

Interactive image sequence player with heatmap overlay for badminton training datasets.
Automatically scans match folders for inputs and heatmaps subdirectories, matches
corresponding image sequences, and displays heatmaps transparently overlaid on original images.

Usage Examples:
    python data_visualizer.py --source /path/to/match1     # Basic: default FPS and alpha
    python data_visualizer.py --source /path/to/match1 --fps 15     # Custom playback rate
    python data_visualizer.py --source /path/to/match1 --alpha 0.4     # Custom transparency

Dependencies:
    pip install opencv-python numpy
"""

import argparse
import csv
import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class SequencePlayer:
    def __init__(self, match_path: str, alpha: float = 0.3):
        self.match_path = Path(match_path)
        self.current_sequence_index = 0
        self.sequence_folders = []
        self.fps = 30  # Default frame rate
        self.alpha = alpha  # Heatmap transparency
        self.show_original_only = False  # Whether to show only original images

    def scan_sequence_folders(self) -> List[str]:
        """Scan all image sequence folders."""
        inputs_path = self.match_path / "inputs"
        heatmaps_path = self.match_path / "heatmaps"

        if not inputs_path.exists() or not heatmaps_path.exists():
            print(f"❌ inputs or heatmaps folder not found in {self.match_path}")
            return []

        # Get all sequence folders from inputs
        sequence_folders = []
        for folder in inputs_path.iterdir():
            if folder.is_dir():
                # Check if corresponding heatmaps folder exists
                corresponding_heatmap = heatmaps_path / folder.name
                if corresponding_heatmap.exists():
                    sequence_folders.append(folder.name)
                else:
                    print(f"⚠️  {folder.name} has no corresponding folder in heatmaps")

        sequence_folders.sort()  # Sort by name
        return sequence_folders

    def load_image_sequence(self, folder_path: Path) -> List[np.ndarray]:
        """Load image sequence from folder."""
        images = []

        # Support multiple image formats
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []

        for pattern in image_patterns:
            image_files.extend(glob.glob(str(folder_path / pattern)))

        # Sort by numeric order
        def extract_number(filename):
            try:
                return int(Path(filename).stem)
            except ValueError:
                return float('inf')

        image_files.sort(key=extract_number)

        for img_file in image_files:
            img = cv2.imread(img_file)
            if img is not None:
                images.append(img)
            else:
                print(f"⚠️  Cannot read image {img_file}")

        return images

    def resize_to_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize images to match (using original image size as target)."""
        h1, w1 = img1.shape[:2]  # Original image size
        h2, w2 = img2.shape[:2]  # Heatmap size

        # Resize heatmap to match original image size
        img2_resized = cv2.resize(img2, (w1, h1))

        return img1, img2_resized

    def apply_colormap_to_heatmap(self, heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Apply color mapping to heatmap."""
        # Convert to grayscale if color image
        if len(heatmap.shape) == 3:
            heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        else:
            heatmap_gray = heatmap

        # Normalize to 0-255 range
        heatmap_norm = cv2.normalize(heatmap_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply color mapping
        heatmap_colored = cv2.applyColorMap(heatmap_norm, colormap)

        return heatmap_colored

    def overlay_images(self, input_img: np.ndarray, heatmap_img: np.ndarray,
                       alpha: float = None) -> np.ndarray:
        """Overlay heatmap on original image."""
        if alpha is None:
            alpha = self.alpha

        # Resize images to match
        input_resized, heatmap_resized = self.resize_to_match(input_img, heatmap_img)

        # If showing original only
        if self.show_original_only:
            return input_resized

        # Apply color mapping to heatmap
        heatmap_colored = self.apply_colormap_to_heatmap(heatmap_resized)

        # Use alpha blending for overlay
        # result = (1-alpha) * input + alpha * heatmap
        overlayed = cv2.addWeighted(input_resized, 1 - alpha, heatmap_colored, alpha, 0)

        return overlayed

    def play_sequence(self, sequence_name: str):
        """Play a single image sequence."""
        inputs_path = self.match_path / "inputs" / sequence_name
        heatmaps_path = self.match_path / "heatmaps" / sequence_name

        print(f"🏸 Loading sequence: {sequence_name}")

        # Load image sequences
        input_images = self.load_image_sequence(inputs_path)
        heatmap_images = self.load_image_sequence(heatmaps_path)

        if not input_images or not heatmap_images:
            print(f"❌ Cannot load image sequence for {sequence_name}")
            return False

        # Ensure both sequences have same length
        min_length = min(len(input_images), len(heatmap_images))
        if len(input_images) != len(heatmap_images):
            print(
                f"⚠️  inputs({len(input_images)}) and heatmaps({len(heatmap_images)}) have different counts, using shorter sequence")

        print(f"🎬 Playing {min_length} frames")

        # Create window
        window_name = f"Heatmap Overlay Player - {sequence_name} ({self.current_sequence_index + 1}/{len(self.sequence_folders)})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_index = 0
        paused = False

        while frame_index < min_length:
            if not paused:
                # Overlay images
                combined_frame = self.overlay_images(
                    input_images[frame_index],
                    heatmap_images[frame_index]
                )

                # Add information text
                alpha_text = "Original Only" if self.show_original_only else f"Alpha: {self.alpha:.2f}"
                info_text = f"Frame: {frame_index + 1}/{min_length} | {alpha_text} | Sequence: {sequence_name}"

                # Add text background for better readability
                text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(combined_frame, (5, 5), (text_size[0] + 15, 35), (0, 0, 0), -1)
                cv2.putText(combined_frame, info_text, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow(window_name, combined_frame)
                frame_index += 1

            # Key handling
            key = cv2.waitKey(int(1000 / self.fps)) & 0xFF

            if key == ord('q') or key == 27:  # q or ESC to exit
                cv2.destroyWindow(window_name)
                return False
            elif key == ord(' '):  # Space to pause/continue
                paused = not paused
                print("⏸️  Paused" if paused else "▶️  Resumed")
            elif key == ord('n') or key == ord('.'):  # n or . for next sequence
                cv2.destroyWindow(window_name)
                return True
            elif key == ord('p') or key == ord(','):  # p or , for previous sequence
                cv2.destroyWindow(window_name)
                return "previous"
            elif key == ord('r'):  # r to restart current sequence
                frame_index = 0
                paused = False
            elif key == ord('o'):  # o to toggle display mode (overlay/original only)
                self.show_original_only = not self.show_original_only
                print(f"🖼️  Display mode: {'Original only' if self.show_original_only else 'Heatmap overlay'}")
            elif key == ord('s'):  # s to save current frame
                save_path = f"frame_{sequence_name}_{frame_index}.jpg"
                cv2.imwrite(save_path, combined_frame)
                print(f"💾 Saved frame to: {save_path}")
            elif key == ord('f'):  # f to fast forward
                frame_index = min(frame_index + 10, min_length - 1)
            elif key == ord('b'):  # b to fast backward
                frame_index = max(frame_index - 10, 0)
            elif key == ord('+') or key == ord('='):  # + to increase transparency
                self.alpha = min(1.0, self.alpha + 0.05)
                print(f"🔆 Alpha: {self.alpha:.2f}")
            elif key == ord('-') or key == ord('_'):  # - to decrease transparency
                self.alpha = max(0.0, self.alpha - 0.05)
                print(f"🔅 Alpha: {self.alpha:.2f}")

        # Sequence finished
        print(f"✅ Sequence {sequence_name} completed")
        cv2.destroyWindow(window_name)
        return True

    def run(self):
        """Run the player."""
        if not self.match_path.exists():
            print(f"❌ Path does not exist: {self.match_path}")
            return

        print(f"🔍 Scanning folder: {self.match_path}")
        self.sequence_folders = self.scan_sequence_folders()

        if not self.sequence_folders:
            print("❌ No valid image sequence folders found")
            return

        print(f"🏸 Found {len(self.sequence_folders)} image sequence folders:")
        for i, folder in enumerate(self.sequence_folders):
            print(f"  {i + 1}. {folder}")

        print(f"\n🎯 Initial alpha: {self.alpha:.2f}")
        print("\n🎮 Controls:")
        print("  Space: Pause/Resume")
        print("  n or .: Next sequence")
        print("  p or ,: Previous sequence")
        print("  r: Restart current sequence")
        print("  o: Toggle display mode (overlay/original only)")
        print("  + or =: Increase heatmap transparency")
        print("  - or _: Decrease heatmap transparency")
        print("  s: Save current frame")
        print("  f: Fast forward 10 frames")
        print("  b: Fast backward 10 frames")
        print("  q or ESC: Exit")
        print()

        # Play all sequences
        while self.current_sequence_index < len(self.sequence_folders):
            sequence_name = self.sequence_folders[self.current_sequence_index]
            result = self.play_sequence(sequence_name)

            if result is False:  # User exit
                break
            elif result == "previous":  # Previous sequence
                self.current_sequence_index = max(0, self.current_sequence_index - 1)
            else:  # Next sequence
                self.current_sequence_index += 1

        cv2.destroyAllWindows()
        print("🏸 Player exited")


class GridSequencePlayer:
    def __init__(self, match_path: str, grid_rows: int = 27, grid_cols: int = 48):
        self.match_path = Path(match_path)
        self.current_sequence_index = 0
        self.sequence_names = []
        self.fps = 30
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.show_grid = True

    def scan_sequences(self) -> List[str]:
        inputs_path = self.match_path / "inputs"
        annotations_path = self.match_path / "annotations"

        if not inputs_path.exists() or not annotations_path.exists():
            print(f"❌ inputs or annotations folder not found in {self.match_path}")
            return []

        sequence_names = []
        for annotation_file in sorted(annotations_path.glob("*.csv")):
            sequence_name = annotation_file.stem
            if (inputs_path / sequence_name).is_dir():
                sequence_names.append(sequence_name)
            else:
                print(f"⚠️  {sequence_name} has no corresponding folder in inputs")

        return sequence_names

    def load_rows(self, annotation_path: Path) -> List[dict]:
        rows = []
        with annotation_path.open(newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append(
                    {
                        "frame": int(row["Frame"]),
                        "visibility": int(row["Visibility"]),
                        "x": float(row["X"]),
                        "y": float(row["Y"]),
                    }
                )
        return rows

    def load_image_sequence(self, folder_path: Path) -> List[np.ndarray]:
        images = []
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_files = []

        for pattern in image_patterns:
            image_files.extend(glob.glob(str(folder_path / pattern)))

        image_files.sort(key=lambda filename: int(Path(filename).stem))

        for img_file in image_files:
            img = cv2.imread(img_file)
            if img is not None:
                images.append(img)
            else:
                print(f"⚠️  Cannot read image {img_file}")

        return images

    def draw_grid_overlay(self, image: np.ndarray, row: dict) -> np.ndarray:
        frame = image.copy()
        height, width = frame.shape[:2]
        cell_w = width / self.grid_cols
        cell_h = height / self.grid_rows

        if self.show_grid:
            for col in range(1, self.grid_cols):
                x = int(round(col * cell_w))
                cv2.line(frame, (x, 0), (x, height - 1), (64, 64, 64), 1)
            for grid_row in range(1, self.grid_rows):
                y = int(round(grid_row * cell_h))
                cv2.line(frame, (0, y), (width - 1, y), (64, 64, 64), 1)

        if row["visibility"]:
            x = float(np.clip(row["x"], 0, width - 1))
            y = float(np.clip(row["y"], 0, height - 1))
            col_idx = min(int(x / cell_w), self.grid_cols - 1)
            row_idx = min(int(y / cell_h), self.grid_rows - 1)

            x0 = int(round(col_idx * cell_w))
            y0 = int(round(row_idx * cell_h))
            x1 = int(round((col_idx + 1) * cell_w))
            y1 = int(round((row_idx + 1) * cell_h))

            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), -1)
            frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
            cv2.circle(frame, (int(round(x)), int(round(y))), 7, (0, 0, 255), -1)
            cv2.circle(frame, (int(round(x)), int(round(y))), 11, (255, 255, 255), 2)

        return frame

    def play_sequence(self, sequence_name: str):
        inputs_path = self.match_path / "inputs" / sequence_name
        annotation_path = self.match_path / "annotations" / f"{sequence_name}.csv"

        print(f"🏸 Loading grid sequence: {sequence_name}")

        input_images = self.load_image_sequence(inputs_path)
        rows = self.load_rows(annotation_path)

        if not input_images or not rows:
            print(f"❌ Cannot load image sequence for {sequence_name}")
            return False

        min_length = min(len(input_images), len(rows))
        if len(input_images) != len(rows):
            print(
                f"⚠️  inputs({len(input_images)}) and annotations({len(rows)}) have different counts, using shorter sequence"
            )

        window_name = (
            f"Grid Dataset Player - {sequence_name} "
            f"({self.current_sequence_index + 1}/{len(self.sequence_names)})"
        )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_index = 0
        paused = False

        while frame_index < min_length:
            if not paused:
                row = rows[frame_index]
                combined_frame = self.draw_grid_overlay(input_images[frame_index], row)
                visibility_text = (
                    f"Visible ({row['x']:.1f}, {row['y']:.1f})"
                    if row["visibility"]
                    else "Not visible"
                )
                info_text = (
                    f"Frame: {frame_index + 1}/{min_length} | "
                    f"Grid: {self.grid_cols}x{self.grid_rows} | {visibility_text}"
                )
                text_size = cv2.getTextSize(
                    info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )[0]
                cv2.rectangle(
                    combined_frame, (5, 5), (text_size[0] + 15, 35), (0, 0, 0), -1
                )
                cv2.putText(
                    combined_frame,
                    info_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow(window_name, combined_frame)
                frame_index += 1

            key = cv2.waitKey(int(1000 / self.fps)) & 0xFF

            if key == ord("q") or key == 27:
                cv2.destroyWindow(window_name)
                return False
            if key == ord(" "):
                paused = not paused
                print("⏸️  Paused" if paused else "▶️  Resumed")
            elif key == ord("n") or key == ord("."):
                cv2.destroyWindow(window_name)
                return True
            elif key == ord("p") or key == ord(","):
                cv2.destroyWindow(window_name)
                return "previous"
            elif key == ord("r"):
                frame_index = 0
                paused = False
            elif key == ord("g"):
                self.show_grid = not self.show_grid
                print(f"🔲 Grid: {'on' if self.show_grid else 'off'}")
            elif key == ord("s"):
                save_path = f"frame_{sequence_name}_{frame_index}.jpg"
                cv2.imwrite(save_path, combined_frame)
                print(f"💾 Saved frame to: {save_path}")
            elif key == ord("f"):
                frame_index = min(frame_index + 10, min_length - 1)
            elif key == ord("b"):
                frame_index = max(frame_index - 10, 0)

        print(f"✅ Sequence {sequence_name} completed")
        cv2.destroyWindow(window_name)
        return True

    def run(self):
        if not self.match_path.exists():
            print(f"❌ Path does not exist: {self.match_path}")
            return

        print(f"🔍 Scanning grid folder: {self.match_path}")
        self.sequence_names = self.scan_sequences()

        if not self.sequence_names:
            print("❌ No valid grid sequences found")
            return

        print(f"🏸 Found {len(self.sequence_names)} grid sequences:")
        for i, folder in enumerate(self.sequence_names):
            print(f"  {i + 1}. {folder}")

        print("\n🎮 Controls:")
        print("  Space: Pause/Resume")
        print("  n or .: Next sequence")
        print("  p or ,: Previous sequence")
        print("  r: Restart current sequence")
        print("  g: Toggle grid lines")
        print("  s: Save current frame")
        print("  f: Fast forward 10 frames")
        print("  b: Fast backward 10 frames")
        print("  q or ESC: Exit")
        print()

        while self.current_sequence_index < len(self.sequence_names):
            sequence_name = self.sequence_names[self.current_sequence_index]
            result = self.play_sequence(sequence_name)

            if result is False:
                break
            if result == "previous":
                self.current_sequence_index = max(0, self.current_sequence_index - 1)
            else:
                self.current_sequence_index += 1

        cv2.destroyAllWindows()
        print("🏸 Grid player exited")


def main():
    """Main function - handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive dataset player for heatmap and grid datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Heatmap Dataset Structure:
    match1/
    ├── inputs/
    │   ├── 1_05_03/     # Sequence identifier (any naming)
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   └── 2_10_07/
    │       └── ...
    └── heatmaps/
        ├── 1_05_03/     # Must correspond to inputs folder names
        │   ├── 0.jpg
        │   ├── 1.jpg
        │   └── ...
        └── 2_10_07/
            └── ...

Grid Dataset Structure:
    match1/
    ├── inputs/
    │   └── sequence_0001/
    │       ├── 1.png
    │       ├── 2.png
    │       └── ...
    └── annotations/
        └── sequence_0001.csv
        """
    )

    parser.add_argument(
        "--source",
        help="Heatmap match folder path containing inputs and heatmaps"
    )

    parser.add_argument(
        "--grid",
        help="Grid match folder path containing inputs and annotations"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback frame rate (default: 30)"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Heatmap transparency (0.0-1.0, default: 0.3)"
    )

    parser.add_argument(
        "--grid-rows",
        type=int,
        default=27,
        help="Grid row count for --grid mode (default: 27)"
    )

    parser.add_argument(
        "--grid-cols",
        type=int,
        default=48,
        help="Grid column count for --grid mode (default: 48)"
    )

    args = parser.parse_args()

    if bool(args.source) == bool(args.grid):
        print("❌ Specify exactly one of --source or --grid")
        return

    if not (0.0 <= args.alpha <= 1.0):
        print("❌ Alpha value must be between 0.0 and 1.0")
        return

    print("🏸 Dataset Training Player")
    print("=" * 50)

    if args.grid:
        player = GridSequencePlayer(
            args.grid, grid_rows=args.grid_rows, grid_cols=args.grid_cols
        )
    else:
        player = SequencePlayer(args.source, args.alpha)
    player.fps = args.fps
    player.run()


if __name__ == "__main__":
    main()
