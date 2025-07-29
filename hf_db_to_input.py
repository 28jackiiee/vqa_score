#!/usr/bin/env python3
"""
Convert a Hugging Face dataset into an input.json of:
[
  {"video": "<url>", "label": "<label>"},
  ...
]

Key flags:
  -database      HF dataset path (e.g. jackieyayqli/vqascore)
  -data-dir      Subdirectory to load/filter from inside the repo (passed to load_dataset)
  -directory     Subdirectory to prepend in constructed URLs (if different from --data-dir)
  -label         Default label used when no label column is present
  --change_label Overwrite all labels in an existing JSON (and exit)
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset

def get_video_url_from_hf(
    video_item: Any,
    dataset_name: str = "",
    file_index: int = 0,
    data_dir: Optional[str] = None,
    directory: Optional[str] = None
) -> str:
    """
    Try to recover or build the original HF URL for a video.

    If the item already contains an http(s) URL, return it.
    Otherwise, build: https://huggingface.co/datasets/{dataset_name}/resolve/main/{subdir}/{filename}
    where subdir = --directory if provided, else --data-dir if provided.
    """
    # Quick exit: plain URL string
    if isinstance(video_item, str) and video_item.startswith(("http://", "https://")):
        return video_item

    filename = None

    # HF Video object with _hf_encoded dict
    if hasattr(video_item, "_hf_encoded") and isinstance(video_item._hf_encoded, dict):
        hf_path = video_item._hf_encoded.get("path", "")
        if hf_path.startswith("hf://"):
            # hf://datasets/user/dset@hash/filename.mp4
            parts = hf_path.replace("hf://", "").split("/")
            if len(parts) >= 3:
                dataset_path = "/".join(parts[1:-1])  # skip 'datasets' + filename
                filename = parts[-1]
                if "@" in dataset_path:
                    dataset_path = dataset_path.split("@")[0]
                subdir = directory or data_dir
                if subdir:
                    return f"https://huggingface.co/datasets/{dataset_path}/resolve/main/{subdir}/{filename}"
                return f"https://huggingface.co/datasets/{dataset_path}/resolve/main/{filename}"
        elif hf_path:
            filename = os.path.basename(hf_path)

    # TorchVision / PyAV style
    elif hasattr(video_item, "container") and hasattr(video_item, "_hf_encoded"):
        try:
            if hasattr(video_item.container, "name"):
                filename = os.path.basename(video_item.container.name)
            elif hasattr(video_item._c, "file") and hasattr(video_item._c.file, "name"):
                filename = os.path.basename(video_item._c.file.name)
        except Exception:
            filename = f"video_{file_index}.mp4"

    # Common attributes
    elif hasattr(video_item, "path"):
        filename = os.path.basename(video_item.path)
    elif hasattr(video_item, "filename"):
        filename = video_item.filename
    elif hasattr(video_item, "name"):
        filename = video_item.name
    elif hasattr(video_item, "url"):
        if isinstance(video_item.url, str) and video_item.url.startswith(("http://", "https://")):
            return video_item.url
        filename = os.path.basename(str(video_item.url))
    elif hasattr(video_item, "src"):
        if isinstance(video_item.src, str) and video_item.src.startswith(("http://", "https://")):
            return video_item.src
        filename = os.path.basename(str(video_item.src))
    elif isinstance(video_item, str):
        if video_item.startswith("/"):
            filename = os.path.basename(video_item)
        else:
            filename = video_item
    elif isinstance(video_item, dict):
        for key in ["url", "src", "path", "filename", "name"]:
            if key in video_item:
                val = video_item[key]
                if isinstance(val, str) and val.startswith(("http://", "https://")):
                    return val
                filename = os.path.basename(str(val))
                break
    else:
        s = str(video_item)
        filename = os.path.basename(s) if "/" in s else s

    # Build HF URL
    if filename and dataset_name and filename != "None" and "<none>" not in str(filename).lower():
        subdir = directory or data_dir
        if subdir:
            return f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{subdir}/{filename}"
        return f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{filename}"

    return filename if filename else f"video_{file_index}.mp4"


def extract_videos_from_dataset(
    dataset: Dataset,
    dataset_name: str = "",
    video_column: str = "video",
    label_column: Optional[str] = None,
    split: str = "train",
    max_items: Optional[int] = None,
    default_label: str = "default_label",
    change_label: Optional[str] = None,
    data_dir: Optional[str] = None,
    directory: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Iterate through the dataset split and build the list of {video, label}.
    """
    results: List[Dict[str, str]] = []

    # Grab the split
    if isinstance(dataset, dict):
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(dataset.keys())}")
        data = dataset[split]
    else:
        data = dataset

    # Column sanity checks for non-streaming
    if hasattr(data, "column_names"):
        if video_column not in data.column_names:
            raise ValueError(f"Video column '{video_column}' not found. Columns: {data.column_names}")
        if label_column and label_column not in data.column_names:
            print(f"Warning: label column '{label_column}' not in columns. Ignoring.")
            label_column = None

    if hasattr(data, "__len__"):
        print(f"Processing {len(data)} items...")
    else:
        print("Processing streaming dataset...")

    override_label = change_label if change_label else None

    for i, item in enumerate(data):
        if max_items and i >= max_items:
            print(f"Reached max-items={max_items}")
            break

        try:
            video_data = item[video_column]
            video_url = get_video_url_from_hf(
                video_item=video_data,
                dataset_name=dataset_name,
                file_index=i,
                data_dir=data_dir,
                directory=directory
            )

            if override_label:
                label = override_label
            elif label_column and label_column in item:
                label = str(item[label_column])
            else:
                label = default_label

            results.append({"video": video_url, "label": label})

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} items...")

        except Exception as e:
            print(f"Error on item {i}: {e}")
            continue

    return results


def change_labels_in_file(path: str, new_label: str) -> int:
    with open(path, "r") as f:
        data = json.load(f)
    count = 0
    for item in data:
        if isinstance(item, dict) and "label" in item:
            item["label"] = new_label
            count += 1
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert HF dataset to input.json")
    parser.add_argument("-database", "--database", help="HF dataset path (e.g. user/dset)")
    parser.add_argument("-data-dir", "--data-dir", default=None,
                        help="Directory in the repo to *load* from (passed to load_dataset)")
    parser.add_argument("-directory", "--directory", default=None,
                        help="Directory to prepend in constructed URLs (if you want a different one)")
    parser.add_argument("-o", "--output", default="input.json", help="Output JSON file")
    parser.add_argument("-label", "--label",
                        default="cam_motion.dolly_zoom_movement.has_dolly_in_zoom_out",
                        help="Default label if none provided")
    parser.add_argument("--change_label", default=None,
                        help="Overwrite labels in existing JSON and exit")
    parser.add_argument("--video-column", default="video",
                        help="Column name for video objects/paths")
    parser.add_argument("--label-column", default=None,
                        help="Column name for labels in the dataset")
    parser.add_argument("--split", default="train", help="Split to read (default: train)")
    parser.add_argument("--subset", default=None, help="Subset/config name (optional)")
    parser.add_argument("--streaming", action="store_true", help="Force streaming mode")
    parser.add_argument("--max-items", type=int, default=None, help="Limit items (debugging)")
    args = parser.parse_args()

    # Fast path: just change labels
    if args.change_label and os.path.exists(args.output):
        print(f"Changing labels in {args.output} -> {args.change_label}")
        n = change_labels_in_file(args.output, args.change_label)
        print(f"Updated {n} entries.")
        return 0

    # Need a dataset unless we're just changing labels
    if not args.database:
        print("Error: --database is required (unless using --change_label on an existing file).")
        return 1

    dataset_name = args.database
    print(f"Loading dataset: {dataset_name}")
    if args.data_dir:
        print(f"  data-dir filter: {args.data_dir}")
    if args.subset:
        print(f"  subset: {args.subset}")
    if args.directory and args.data_dir and args.directory != args.data_dir:
        print(f"Warning: --data-dir ({args.data_dir}) != --directory ({args.directory}).")

    try:
        dataset = load_dataset(
            dataset_name,
            name=args.subset,
            data_dir=args.data_dir if args.data_dir else None,
            streaming=args.streaming or True  # streaming=True ensures URLs not local tmp files
        )
        print("Dataset loaded.")
        if isinstance(dataset, dict):
            print(f"Available splits: {list(dataset.keys())}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    json_data = extract_videos_from_dataset(
        dataset=dataset,
        dataset_name=dataset_name,
        video_column=args.video_column,
        label_column=args.label_column,
        split=args.split,
        max_items=args.max_items,
        default_label=args.label,
        change_label=args.change_label,
        data_dir=args.data_dir,
        directory=args.directory
    )

    if not json_data:
        print("No entries extracted.")
        return 1

    with open(args.output, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"\nJSON written to {args.output} ({len(json_data)} entries)")
    uniq = set(x["label"] for x in json_data)
    if len(uniq) > 1:
        print("Label distribution:")
        for lab in sorted(uniq):
            c = sum(1 for x in json_data if x["label"] == lab)
            print(f"  {lab}: {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())