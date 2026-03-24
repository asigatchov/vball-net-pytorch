# convert_v2_to_v1b.py

import torch
import os
import argparse
import re


def convert_v2_checkpoint_to_v1b(input_path: str, output_path: str = None):
    """
    Convert a VballNetV2 checkpoint into a format compatible with VballNetV1b.
    Replaces the key prefix in state_dict: 'VballNetV2.' -> 'VballNetV1b.'

    Parameters:
        input_path (str): path to the source .pth file (VballNetV2)
        output_path (str): path to save the converted file.
                           If None, appends the '_v1b' suffix to the source name.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')

    # Detect state_dict depending on checkpoint structure
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            checkpoint_type = 'with_wrapper'
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            checkpoint_type = 'with_wrapper'
        else:
            state_dict = checkpoint
            checkpoint_type = 'raw'
    else:
        state_dict = checkpoint
        checkpoint_type = 'raw'

    print(f"Checkpoint type: {checkpoint_type}")
    print(f"Number of parameters before conversion: {len(state_dict)}")

    # Count and replace prefixes
    converted_count = 0
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('VballNetV2.'):
            new_key = 'VballNetV1b.' + key[len('VballNetV2.'):]
            converted_count += 1
        else:
            new_key = key  # keep unchanged (for example optimizer, epoch, etc.)
        new_state_dict[new_key] = value

    print(f"Prefixes replaced: {converted_count}")

    # Restore checkpoint structure
    if checkpoint_type == 'with_wrapper':
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = new_state_dict
        elif 'model_state_dict' in checkpoint:
            checkpoint['model_state_dict'] = new_state_dict
        converted_checkpoint = checkpoint
    else:
        converted_checkpoint = new_state_dict

    # Determine the output path
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        # Automatically detect seq if it is present in the filename
        seq_match = re.search(r'seq(\d+)', base)
        seq_suffix = f"_seq{seq_match.group(1)}" if seq_match else ""
        output_path = f"{base}{seq_suffix}_v1b{ext}"

    print(f"Saving converted checkpoint: {output_path}")
    torch.save(converted_checkpoint, output_path)
    print("Conversion completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a VballNetV2 checkpoint to VballNetV1b format (rename class prefix in state_dict)"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Path to the source .pth file (VballNetV2)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Path to save the converted file. If omitted, the '_v1b' suffix is added"
    )

    args = parser.parse_args()

    convert_v2_checkpoint_to_v1b(args.input, args.output)
