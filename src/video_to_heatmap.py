"""
Badminton Dataset Preprocessor

Processes raw badminton video datasets for machine learning training.
Extracts video frames, resizes to 512×288 resolution, and generates Gaussian
heatmaps for shuttlecock position detection in a streamlined pipeline.

Usage Examples:
    python video_to_heatmap.py --source dataset --output dataset_preprocessed
    python video_to_heatmap.py --source /path/to/data --output /path/to/output --sigma 5
    python video_to_heatmap.py --source dataset --sigma 4 --force
    python video_to_heatmap.py --source dataset --frame_step 2  # Process every 2nd frame

Dependencies:
    pip install opencv-python pandas numpy scipy tqdm
"""

import argparse
import os
import shutil
import sys
import gc
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm

# System files to ignore during processing
IGNORED_FILES = {'.DS_Store', 'Thumbs.db', '.gitignore', '.gitkeep'}
IGNORED_DIRS = {'.git', '__pycache__', '.vscode', '.idea', 'node_modules'}

# Target dimensions for processed images
TARGET_WIDTH = 512
TARGET_HEIGHT = 288
GRID_TARGET_WIDTH = 768
GRID_TARGET_HEIGHT = 432

# Форматы сохранения
FORMAT_PNG_GRAYSCALE = 'PNG_GRAYSCALE'
DEFAULT_IMAGE_FORMAT = FORMAT_PNG_GRAYSCALE
JPEG_QUALITY = 95


def is_valid_path(name):
    """Check if file or directory name should be processed."""
    if name.startswith('.') and name not in {'.', '..'}:
        return False
    return name not in IGNORED_FILES and name not in IGNORED_DIRS


def generate_heatmap(center_x, center_y, width=TARGET_WIDTH, height=TARGET_HEIGHT, sigma=3):
    """Generate 2D Gaussian heatmap centered at specified coordinates."""
    x_coords = np.arange(0, width)
    y_coords = np.arange(0, height)
    mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
    coordinates = np.dstack((mesh_x, mesh_y))

    gaussian_mean = [center_x, center_y]
    covariance_matrix = [[sigma ** 2, 0], [0, sigma ** 2]]

    distribution = multivariate_normal(gaussian_mean, covariance_matrix)
    heatmap = distribution.pdf(coordinates)

    # Normalize to 0-255 range
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)

    return heatmap_uint8


def resize_with_aspect_ratio(image, target_w=TARGET_WIDTH, target_h=TARGET_HEIGHT):
    """Resize image maintaining aspect ratio and center on target canvas."""
    original_h, original_w = image.shape[:2]

    # Calculate scaling factor
    scale_width = target_w / original_w
    scale_height = target_h / original_h
    scale_factor = min(scale_width, scale_height)

    # Calculate new dimensions
    new_width = int(original_w * scale_factor)
    new_height = int(original_h * scale_factor)

    # Resize image
    resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create target canvas and center the image
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    offset_x = (target_w - new_width) // 2
    offset_y = (target_h - new_height) // 2
    canvas[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized_img

    return canvas, scale_factor, offset_x, offset_y


def resize_stretch(image, target_w, target_h):
    """Resize image with direct stretching."""
    original_h, original_w = image.shape[:2]
    resized_img = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    scale_x = target_w / original_w
    scale_y = target_h / original_h
    return resized_img, scale_x, scale_y


def transform_annotation_coords(x, y, scale, offset_x, offset_y):
    """Transform annotation coordinates based on image resizing parameters."""
    transformed_x = x * scale + offset_x
    transformed_y = y * scale + offset_y
    return transformed_x, transformed_y


def transform_annotation_coords_stretch(x, y, scale_x, scale_y):
    return x * scale_x, y * scale_y


def validate_dataset_structure(source_path):
    """Validate source dataset structure and return summary statistics."""
    if not os.path.exists(source_path):
        return False, f"Source path does not exist: {source_path}"

    # Discover match directories
    entries = [item for item in os.listdir(source_path) if is_valid_path(item)]
    match_dirs = [
        item for item in entries
        if item.startswith("match") and os.path.isdir(os.path.join(source_path, item))
    ]

    if not match_dirs:
        return False, "No match directories found"

    # Analyze structure
    valid_matches = 0
    video_count = 0
    annotation_count = 0

    for match_dir in match_dirs:
        match_path = os.path.join(source_path, match_dir)
        annotations_path = os.path.join(match_path, "csv")
        videos_path = os.path.join(match_path, "video")

        if os.path.exists(annotations_path) and os.path.exists(videos_path):
            valid_matches += 1

            # Count annotation files
            csv_files = [f for f in os.listdir(annotations_path)
                         if f.endswith('_ball.csv') and is_valid_path(f)]
            annotation_count += len(csv_files)

            # Count video files
            mp4_files = [f for f in os.listdir(videos_path)
                         if f.endswith('.mp4') and is_valid_path(f)]
            video_count += len(mp4_files)

    if valid_matches == 0:
        return False, "No valid match directories found (must contain both csv and video subdirectories)"

    summary = f"Found {valid_matches} match directories, {video_count} videos, {annotation_count} annotation files"
    return True, summary


def estimate_total_frames(source_path):
    """Estimate total number of frames across all videos for progress tracking."""
    frame_total = 0
    match_dirs = [
        item for item in os.listdir(source_path) if is_valid_path(item)
                                                    and item.startswith("match") and os.path.isdir(
            os.path.join(source_path, item))
    ]

    for match_dir in match_dirs:
        match_path = os.path.join(source_path, match_dir)
        videos_path = os.path.join(match_path, "video")

        if os.path.exists(videos_path):
            mp4_files = [f for f in os.listdir(videos_path)
                         if f.endswith('.mp4') and is_valid_path(f)]

            for mp4_file in mp4_files:
                video_path = os.path.join(videos_path, mp4_file)
                video_capture = cv2.VideoCapture(video_path)
                if video_capture.isOpened():
                    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_total += total_frames
                video_capture.release()

    return frame_total


def process_video_sequence(video_path, annotation_path, inputs_output_dir, heatmaps_output_dir,
                           sequence_name, sigma_value, frame_step, progress_bar):
    """Process single video sequence with corresponding annotations."""

    # Create output directories
    sequence_inputs_dir = os.path.join(inputs_output_dir, sequence_name)
    sequence_heatmaps_dir = os.path.join(heatmaps_output_dir, sequence_name)
    os.makedirs(sequence_inputs_dir, exist_ok=True)
    os.makedirs(sequence_heatmaps_dir, exist_ok=True)

    # Load annotations
    if not os.path.exists(annotation_path):
        tqdm.write(f"Warning: Annotation file not found {annotation_path}")
        return 0

    try:
        annotations_df = pd.read_csv(annotation_path)
    except Exception as e:
        tqdm.write(f"Error: Cannot read annotation file {annotation_path}: {e}")
        return 0

    # Open video stream
    video_stream = cv2.VideoCapture(video_path)
    if not video_stream.isOpened():
        tqdm.write(f"Error: Cannot open video {video_path}")
        return 0

    frames_processed = 0
    current_frame = 0
    output_frame_index = 0  # Continuous frame numbering regardless of frame_step
    encoding_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    image_format = DEFAULT_IMAGE_FORMAT

    # Create frame annotation lookup for efficiency
    annotation_lookup = {}
    for _, row in annotations_df.iterrows():
        annotation_lookup[row['Frame']] = row

    try:
        while True:
            frame_available, frame_data = video_stream.read()
            if not frame_available:
                break

            # Update progress
            progress_bar.update(1)

            # Process frame if annotation exists and it's selected by frame_step
            if current_frame in annotation_lookup and current_frame % frame_step == 0:
                annotation_row = annotation_lookup[current_frame]

                # Resize frame
                processed_frame, scale_factor, x_offset, y_offset = resize_with_aspect_ratio(frame_data)

                # Generate heatmap based on visibility
                if annotation_row['Visibility'] == 1:
                    # Visible shuttlecock: create heatmap at annotation position
                    original_x = annotation_row['X']
                    original_y = annotation_row['Y']

                    if pd.isna(original_x) or pd.isna(original_y):
                        heatmap = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
                    else:
                        # Transform coordinates to resized image space
                        transformed_x, transformed_y = transform_annotation_coords(
                            original_x, original_y, scale_factor, x_offset, y_offset
                        )

                        # Clamp coordinates to image bounds
                        transformed_x = max(0, min(TARGET_WIDTH - 1, transformed_x))
                        transformed_y = max(0, min(TARGET_HEIGHT - 1, transformed_y))

                        # Generate Gaussian heatmap
                        heatmap = generate_heatmap(transformed_x, transformed_y, sigma=sigma_value)
                else:
                    # Invisible shuttlecock: zero heatmap
                    heatmap = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)


                # Сохраняем кадры и тепловые карты в PNG (градации серого)
                # Use continuous numbering for output frames
                frame_output_path = os.path.join(sequence_inputs_dir, f"{output_frame_index}.png")
                heatmap_output_path = os.path.join(sequence_heatmaps_dir, f"{output_frame_index}.png")

                # Кадр переводим в оттенки серого
                processed_frame_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(frame_output_path, processed_frame_gray)
                cv2.imwrite(heatmap_output_path, heatmap)

                frames_processed += 1
                output_frame_index += 1  # Increment output frame counter

                # Memory cleanup
                del processed_frame, heatmap

            current_frame += 1

            # Periodic garbage collection
            if current_frame % 100 == 0:
                gc.collect()

    finally:
        video_stream.release()

    return frames_processed


def process_match_directory(match_path, output_base_dir, sigma_value, frame_step, progress_bar):
    """Process single match directory containing videos and annotations."""
    match_name = os.path.basename(match_path)

    # Setup output structure
    match_output_dir = os.path.join(output_base_dir, match_name)
    inputs_output_dir = os.path.join(match_output_dir, "inputs")
    heatmaps_output_dir = os.path.join(match_output_dir, "heatmaps")

    os.makedirs(inputs_output_dir, exist_ok=True)
    os.makedirs(heatmaps_output_dir, exist_ok=True)

    # Get video and annotation directories
    videos_dir = os.path.join(match_path, "video")
    annotations_dir = os.path.join(match_path, "csv")

    if not os.path.exists(videos_dir) or not os.path.exists(annotations_dir):
        tqdm.write(f"Warning: Missing video or csv directory in {match_name}")
        return

    # Get video files
    mp4_files = [f for f in os.listdir(videos_dir)
                 if f.endswith('.mp4') and is_valid_path(f)]

    sequences_processed = 0
    total_frames_processed = 0

    for mp4_file in mp4_files:
        video_path = os.path.join(videos_dir, mp4_file)
        sequence_name = Path(mp4_file).stem

        # Find corresponding annotation file
        annotation_filename = f"{sequence_name}_ball.csv"
        annotation_path = os.path.join(annotations_dir, annotation_filename)

        if os.path.exists(annotation_path):
            frames_count = process_video_sequence(
                video_path, annotation_path, inputs_output_dir, heatmaps_output_dir,
                sequence_name, sigma_value, frame_step, progress_bar
            )

            if frames_count > 0:
                sequences_processed += 1
                total_frames_processed += frames_count
        else:
            tqdm.write(f"Warning: Annotation file not found for {mp4_file}")

    tqdm.write(f"Completed {match_name}: {sequences_processed} sequences, {total_frames_processed} frames")


def process_grid_video_sequence(
    video_path,
    annotation_path,
    inputs_output_dir,
    annotations_output_dir,
    sequence_name,
    frame_step,
    progress_bar,
):
    sequence_inputs_dir = os.path.join(inputs_output_dir, sequence_name)
    os.makedirs(sequence_inputs_dir, exist_ok=True)
    os.makedirs(annotations_output_dir, exist_ok=True)

    try:
        annotations_df = pd.read_csv(annotation_path)
    except Exception as error:
        tqdm.write(f"Error: Cannot read annotation file {annotation_path}: {error}")
        return 0

    video_stream = cv2.VideoCapture(video_path)
    if not video_stream.isOpened():
        tqdm.write(f"Error: Cannot open video {video_path}")
        return 0

    annotation_lookup = {}
    for _, row in annotations_df.iterrows():
        annotation_lookup[int(row["Frame"])] = row

    transformed_rows = []
    frames_processed = 0
    current_frame = 0
    output_frame_index = 0

    try:
        while True:
            frame_available, frame_data = video_stream.read()
            if not frame_available:
                break

            progress_bar.update(1)

            if current_frame in annotation_lookup and current_frame % frame_step == 0:
                annotation_row = annotation_lookup[current_frame]
                processed_frame, scale_x, scale_y = resize_stretch(
                    frame_data, GRID_TARGET_WIDTH, GRID_TARGET_HEIGHT
                )

                visibility = int(annotation_row["Visibility"])
                if visibility == 1 and not (pd.isna(annotation_row["X"]) or pd.isna(annotation_row["Y"])):
                    transformed_x, transformed_y = transform_annotation_coords_stretch(
                        float(annotation_row["X"]),
                        float(annotation_row["Y"]),
                        scale_x,
                        scale_y,
                    )
                    transformed_x = max(0.0, min(GRID_TARGET_WIDTH - 1, transformed_x))
                    transformed_y = max(0.0, min(GRID_TARGET_HEIGHT - 1, transformed_y))
                else:
                    transformed_x = 0.0
                    transformed_y = 0.0

                frame_output_path = os.path.join(sequence_inputs_dir, f"{output_frame_index}.png")
                cv2.imwrite(frame_output_path, processed_frame)
                transformed_rows.append(
                    {
                        "Frame": output_frame_index,
                        "Visibility": visibility,
                        "X": float(transformed_x),
                        "Y": float(transformed_y),
                    }
                )
                frames_processed += 1
                output_frame_index += 1

            current_frame += 1
            if current_frame % 100 == 0:
                gc.collect()
    finally:
        video_stream.release()

    if transformed_rows:
        annotation_output_path = os.path.join(annotations_output_dir, f"{sequence_name}.csv")
        pd.DataFrame(transformed_rows).to_csv(annotation_output_path, index=False)

    return frames_processed


def process_grid_match_directory(match_path, output_base_dir, frame_step, progress_bar):
    match_name = os.path.basename(match_path)
    match_output_dir = os.path.join(output_base_dir, match_name)
    inputs_output_dir = os.path.join(match_output_dir, "inputs")
    annotations_output_dir = os.path.join(match_output_dir, "annotations")

    os.makedirs(inputs_output_dir, exist_ok=True)
    os.makedirs(annotations_output_dir, exist_ok=True)

    videos_dir = os.path.join(match_path, "video")
    annotations_dir = os.path.join(match_path, "csv")
    if not os.path.exists(videos_dir) or not os.path.exists(annotations_dir):
        tqdm.write(f"Warning: Missing video or csv directory in {match_name}")
        return

    mp4_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4") and is_valid_path(f)]
    sequences_processed = 0
    total_frames_processed = 0

    for mp4_file in mp4_files:
        video_path = os.path.join(videos_dir, mp4_file)
        sequence_name = Path(mp4_file).stem
        annotation_path = os.path.join(annotations_dir, f"{sequence_name}_ball.csv")
        if not os.path.exists(annotation_path):
            tqdm.write(f"Warning: Annotation file not found for {mp4_file}")
            continue

        frames_count = process_grid_video_sequence(
            video_path,
            annotation_path,
            inputs_output_dir,
            annotations_output_dir,
            sequence_name,
            frame_step,
            progress_bar,
        )
        if frames_count > 0:
            sequences_processed += 1
            total_frames_processed += frames_count

    tqdm.write(
        f"Completed {match_name}: {sequences_processed} sequences, {total_frames_processed} frames"
    )


def preprocess_dataset(source_path, output_path, sigma_value=3.0, frame_step=1, force_overwrite=False):
    """Main preprocessing pipeline for badminton dataset."""

    # Validate input structure
    structure_valid, validation_message = validate_dataset_structure(source_path)
    if not structure_valid:
        print(f"❌ {validation_message}")
        return False

    print(f"✅ {validation_message}")

    # Handle existing output directory
    if os.path.exists(output_path):
        if force_overwrite:
            print(f"🗑️ Removing existing directory: {output_path}")
            shutil.rmtree(output_path)
        else:
            user_input = input(f"⚠️ Output directory exists: {output_path}\n   Delete and rebuild? (y/n): ")
            if user_input.lower() != 'y':
                print("❌ Operation cancelled")
                return False
            shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)

    # Discover valid match directories
    entries = [item for item in os.listdir(source_path) if is_valid_path(item)]
    match_dirs = [
        item for item in entries
        if item.startswith("match") and os.path.isdir(os.path.join(source_path, item))
    ]

    # Filter for valid match directories
    valid_match_dirs = []
    for match_dir_name in match_dirs:
        match_dir_path = os.path.join(source_path, match_dir_name)
        has_csv = os.path.exists(os.path.join(match_dir_path, "csv"))
        has_video = os.path.exists(os.path.join(match_dir_path, "video"))

        if has_csv and has_video:
            valid_match_dirs.append(match_dir_name)
        else:
            print(f"⚠️ Skipping {match_dir_name}: missing csv or video directory")

    if not valid_match_dirs:
        print("❌ No valid match directories found")
        return False

    print(f"🚀 Processing {len(valid_match_dirs)} match directories...")

    # Estimate total workload
    print("📊 Estimating workload...")
    total_frame_count = estimate_total_frames(source_path)
    print(f"📊 Total frames to process: {total_frame_count}")

    # Execute preprocessing with progress tracking
    with tqdm(total=total_frame_count // frame_step, desc="Processing frames", unit="frame") as progress_bar:
        for match_dir_name in valid_match_dirs:
            match_dir_path = os.path.join(source_path, match_dir_name)
            process_match_directory(match_dir_path, output_path, sigma_value, frame_step, progress_bar)

            # Memory cleanup after each match
            gc.collect()

    print(f"\n🎉 Preprocessing completed!")
    print(f"   Source: {source_path}")
    print(f"   Output: {output_path}")
    print(f"   Heatmap sigma: {sigma_value}")
    print(f"   Frame step: {frame_step} (process every {frame_step} frame(s))")
    return True


def preprocess_grid_dataset(source_path, output_path, frame_step=1, force_overwrite=False):
    structure_valid, validation_message = validate_dataset_structure(source_path)
    if not structure_valid:
        print(f"ERROR: {validation_message}")
        return False

    print(f"OK: {validation_message}")

    if os.path.exists(output_path):
        if force_overwrite:
            shutil.rmtree(output_path)
        else:
            user_input = input(f"Output directory exists: {output_path}\nDelete and rebuild? (y/n): ")
            if user_input.lower() != "y":
                print("Operation cancelled")
                return False
            shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)

    entries = [item for item in os.listdir(source_path) if is_valid_path(item)]
    valid_match_dirs = [
        item
        for item in entries
        if item.startswith("match")
        and os.path.isdir(os.path.join(source_path, item))
        and os.path.exists(os.path.join(source_path, item, "csv"))
        and os.path.exists(os.path.join(source_path, item, "video"))
    ]

    if not valid_match_dirs:
        print("ERROR: No valid match directories found")
        return False

    total_frame_count = estimate_total_frames(source_path)
    with tqdm(total=total_frame_count // frame_step, desc="Processing frames", unit="frame") as progress_bar:
        for match_dir_name in valid_match_dirs:
            match_dir_path = os.path.join(source_path, match_dir_name)
            process_grid_match_directory(match_dir_path, output_path, frame_step, progress_bar)
            gc.collect()

    print("\nGrid preprocessing completed")
    print(f"   Source: {source_path}")
    print(f"   Output: {output_path}")
    print(f"   Target size: {GRID_TARGET_WIDTH}x{GRID_TARGET_HEIGHT}")
    print(f"   Frame step: {frame_step}")
    return True


def main():
    """Command line interface for dataset preprocessing."""
    parser = argparse.ArgumentParser(
        description="Badminton Dataset Preprocessor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input Structure:
    dataset/
    ├── match1/
    │   ├── csv/
    │   │   └── rally1_ball.csv
    │   └── video/
    │       └── rally1.mp4
    └── match2/...

Output Structure:
    dataset_preprocessed/
    ├── match1/
    │   ├── inputs/rally1/0.jpg,1.jpg... (512×288)
    │   └── heatmaps/rally1/0.jpg,1.jpg... (Gaussian heatmaps)
    └── match2/...

Annotation Format (rally1_ball.csv):
    Frame,Visibility,X,Y
    0,1,637.0,346.0
    1,1,639.0,346.0
    2,0,640.0,345.0  # Visibility=0 generates zero heatmap
        """
    )


    parser.add_argument(
        "--source",
        required=True,
        help="Source dataset directory path"
    )

    parser.add_argument(
        "--output",
        help="Output directory path (default: source + '_preprocessed')"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Gaussian heatmap standard deviation (default: 3.0)"
    )

    parser.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="Frame sampling step (default: 1 - process all frames, 2 - every 2nd frame, etc.)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing output directory"
    )

    parser.add_argument(
        "--mode",
        choices=["heatmap", "grid"],
        default="heatmap",
        help="Output dataset mode"
    )

    # (Опционально) аргумент для формата сохранения
    parser.add_argument(
        "--format",
        choices=[FORMAT_PNG_GRAYSCALE],
        default=DEFAULT_IMAGE_FORMAT,
        help="Output image format (default: PNG_GRAYSCALE)"
    )

    args = parser.parse_args()

    print("🏸 Badminton Dataset Preprocessor")
    print("=" * 50)

    # Verify dependencies
    try:
        print(f"📦 OpenCV {cv2.__version__}")
        print(f"📦 NumPy {np.__version__}")
        print(f"📦 Pandas {pd.__version__}")
    except ImportError as error:
        print(f"❌ Missing dependency: {error}")
        print("Install with: pip install opencv-python pandas numpy scipy tqdm")
        sys.exit(1)

    # Set default output path if not provided
    if not args.output:
        dataset_name = os.path.basename(args.source.rstrip('/'))
        parent_directory = os.path.dirname(args.source) or '.'
        args.output = os.path.join(parent_directory, f"{dataset_name}_preprocessed")

    print(f"📂 Source: {args.source}")
    print(f"📂 Output: {args.output}")
    print(f"🧩 Mode: {args.mode}")
    if args.mode == "heatmap":
        print(f"🎯 Heatmap sigma: {args.sigma}")
    print(f"⏭️  Frame step: {args.frame_step}")


    if args.mode == "grid":
        success = preprocess_grid_dataset(args.source, args.output, args.frame_step, args.force)
    else:
        success = preprocess_dataset(args.source, args.output, args.sigma, args.frame_step, args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
