#!/usr/bin/env python3
"""
Script to convert a Hugging Face dataset into a JSON input file.
Takes a dataset name/path and generates a JSON file with video paths and labels.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset

def get_video_url_from_hf(video_item: Any, dataset_name: str = "", file_index: int = 0) -> str:
    """
    Extract the original Hugging Face URL from video data.
    
    Args:
        video_item: Video data from the dataset (could be string URL, path, or video object)
        dataset_name: Name of the HF dataset for constructing URLs
        file_index: Index for constructing URLs if needed
        
    Returns:
        Original URL or path to the video file
    """
    # If it's already a URL string, return it
    if isinstance(video_item, str) and video_item.startswith(('http://', 'https://')):
        return video_item
    
    # Try to extract filename from various video object types
    filename = None
    
    # Handle VideoReader objects from torchvision/HF datasets
    if hasattr(video_item, '_hf_encoded') and isinstance(video_item._hf_encoded, dict):
        # Extract from _hf_encoded path
        hf_path = video_item._hf_encoded.get('path', '')
        if hf_path.startswith('hf://'):
            # Convert hf:// URL to https:// URL
            # Format: hf://datasets/username/dataset@hash/filename.mp4
            # Convert to: https://huggingface.co/datasets/username/dataset/resolve/main/filename.mp4
            path_parts = hf_path.replace('hf://', '').split('/')
            if len(path_parts) >= 3:
                # Extract dataset path and filename
                dataset_path = '/'.join(path_parts[1:-1])  # Skip 'datasets' and filename
                filename = path_parts[-1]  # Get filename
                
                # Remove the @hash part if present
                if '@' in dataset_path:
                    dataset_path = dataset_path.split('@')[0]
                
                return f"https://huggingface.co/datasets/{dataset_path}/resolve/main/{filename}"
        elif hf_path:
            # Extract filename from path
            filename = os.path.basename(hf_path)
    
    # Handle VideoReader objects with container
    elif hasattr(video_item, 'container') and hasattr(video_item, '_hf_encoded'):
        # This is a VideoReader object from HF datasets
        if hasattr(video_item.container, 'name'):
            path = video_item.container.name
            filename = os.path.basename(path)
        elif hasattr(video_item, '_c') and hasattr(video_item._c, 'file') and hasattr(video_item._c.file, 'name'):
            path = video_item._c.file.name
            filename = os.path.basename(path)
        else:
            # Try to get the source path from the container
            try:
                container = video_item.container
                if hasattr(container, 'name'):
                    filename = os.path.basename(container.name)
                elif hasattr(container, 'metadata'):
                    metadata = container.metadata
                    if 'filename' in metadata:
                        filename = metadata['filename']
                elif hasattr(container, 'file'):
                    if hasattr(container.file, 'name'):
                        filename = os.path.basename(container.file.name)
                    else:
                        filename = str(container.file)
            except Exception as e:
                print(f"Warning: Could not extract filename from VideoReader: {e}")
                filename = f"video_{file_index}.mp4"
    
    # Try other common video object attributes
    elif hasattr(video_item, 'path'):
        filename = os.path.basename(video_item.path)
    elif hasattr(video_item, 'filename'):
        filename = video_item.filename
    elif hasattr(video_item, 'name'):
        filename = video_item.name
    elif hasattr(video_item, 'url'):
        # If there's a URL attribute, return it directly
        return video_item.url
    elif hasattr(video_item, 'src'):
        # Check for src attribute
        if isinstance(video_item.src, str) and video_item.src.startswith(('http://', 'https://')):
            return video_item.src
        else:
            filename = os.path.basename(str(video_item.src))
    elif isinstance(video_item, str):
        if video_item.startswith('/'):
            # It's a local path, extract filename
            filename = os.path.basename(video_item)
        else:
            # Assume it's already a filename or URL
            filename = video_item
    elif isinstance(video_item, dict):
        # Handle dictionary-like video objects
        for key in ['url', 'src', 'path', 'filename', 'name']:
            if key in video_item:
                value = video_item[key]
                if isinstance(value, str):
                    if value.startswith(('http://', 'https://')):
                        return value
                    else:
                        filename = os.path.basename(value)
                        break
    else:
        # Try to convert to string and extract filename
        str_item = str(video_item)
        if '/' in str_item:
            filename = os.path.basename(str_item)
        else:
            filename = str_item
    
    # If we have a dataset name and filename, construct HF URL
    if filename and dataset_name and filename != 'None' and '<none>' not in str(filename).lower():
        # Clean dataset name for URL
        dataset_clean = dataset_name.replace('/', '--')
        # Construct Hugging Face URL
        hf_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{filename}"
        return hf_url
    
    # Fallback to filename or original item
    return filename if filename else f"video_{file_index}.mp4"

def extract_videos_from_dataset(dataset: Dataset, 
                               dataset_name: str = "",
                               video_column: str = "video",
                               label_column: Optional[str] = None,
                               split: str = "train",
                               max_items: Optional[int] = None,
                               default_label: str = "default_label",
                               change_label: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Extract video URLs and labels from a Hugging Face dataset.
    
    Args:
        dataset: Loaded HF dataset
        dataset_name: Name of the HF dataset for URL construction
        video_column: Name of the column containing video data
        label_column: Name of the column containing labels (optional)
        split: Dataset split to use
        max_items: Maximum number of items to process (for streaming datasets)
        default_label: Default label to use when no label column is specified
        change_label: Override label to apply to all videos (optional)
        
    Returns:
        List of dictionaries with video and label keys
    """
    results = []
    
    # Get the specified split
    if isinstance(dataset, dict):
        if split not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(f"Split '{split}' not found. Available splits: {available_splits}")
        data = dataset[split]
    else:
        data = dataset
    
    # Check if required columns exist (for non-streaming datasets)
    if hasattr(data, 'column_names'):
        if video_column not in data.column_names:
            raise ValueError(f"Video column '{video_column}' not found. Available columns: {data.column_names}")
        
        if label_column and label_column not in data.column_names:
            print(f"Warning: Label column '{label_column}' not found. Available columns: {data.column_names}")
            label_column = None
    
    # Handle streaming vs non-streaming datasets
    if hasattr(data, '__len__'):
        print(f"Processing {len(data)} items from dataset...")
    else:
        print("Processing streaming dataset...")
    
    # Determine which label to use
    if change_label:
        print(f"Using override label for all videos: {change_label}")
        use_label = change_label
    else:
        use_label = None  # Will be determined per item
    
    for i, item in enumerate(data):
        try:
            # Check if we've reached max items limit
            if max_items and i >= max_items:
                print(f"Reached maximum items limit: {max_items}")
                break
                
            # Extract video
            video_data = item[video_column]
            video_url = get_video_url_from_hf(video_data, dataset_name, i)
            
            # Extract label
            if use_label:
                # Use the override label
                label = use_label
            elif label_column and label_column in item:
                label = str(item[label_column])
            else:
                label = default_label
            
            results.append({
                "video": video_url,
                "label": label
            })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} items...")
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Convert Hugging Face dataset to JSON input file")
    parser.add_argument("-database", "--database", required=False, help="Hugging Face dataset name or path")
    parser.add_argument("-o", "--output", default="input.json", help="Output JSON file (default: input.json)")
    parser.add_argument("-label", "--label", default="cam_motion.dolly_zoom_movement.has_dolly_in_zoom_out", 
                       help="Default label if no label column specified")
    parser.add_argument("--change_label", default=None, 
                       help="Override label to apply to all videos (ignores dataset labels)")
    parser.add_argument("--video-column", default="video", 
                       help="Name of the video column in the dataset (default: video)")
    parser.add_argument("--label-column", default=None, 
                       help="Name of the label column in the dataset (optional)")
    parser.add_argument("--split", default="train", 
                       help="Dataset split to use (default: train)")
    parser.add_argument("--subset", default=None, 
                       help="Dataset subset/config name (optional)")
    parser.add_argument("--streaming", action="store_true", 
                       help="Use streaming mode for large datasets")
    parser.add_argument("--max-items", type=int, default=None, 
                       help="Maximum number of items to process (optional)")
    
    args = parser.parse_args()
    
    try:
        # Check if --change_label is provided and input.json exists
        if args.change_label and os.path.exists(args.output):
            print(f"Found existing {args.output}, changing labels to: {args.change_label}")
            
            # Load existing JSON file
            with open(args.output, 'r') as f:
                json_data = json.load(f)
            
            # Update all labels
            for item in json_data:
                if isinstance(item, dict) and 'label' in item:
                    item['label'] = args.change_label
            
            # Write back to file
            with open(args.output, 'w') as f:
                json.dump(json_data, f, indent=4)
            
            print(f"Updated {len(json_data)} entries with new label: {args.change_label}")
            print(f"JSON file updated: {args.output}")
            return 0
        
        # If no change_label or no existing file, require database
        if not args.database:
            print("Error: --database is required when not using --change_label on existing file")
            return 1
        
        # Load dataset with streaming to avoid automatic downloads
        print(f"Loading dataset: {args.database}")
        if args.subset:
            print(f"Using subset: {args.subset}")
        
        # Force streaming mode to get URLs instead of downloaded files
        dataset = load_dataset(
            args.database, 
            name=args.subset,
            streaming=True  # Force streaming to avoid downloads
        )
        
        print(f"Dataset loaded successfully!")
        if isinstance(dataset, dict):
            print(f"Available splits: {list(dataset.keys())}")
        
        # Extract videos with original URLs
        json_data = extract_videos_from_dataset(
            dataset=dataset,
            dataset_name=args.database,
            video_column=args.video_column,
            label_column=args.label_column,
            split=args.split,
            max_items=args.max_items,
            default_label=args.label,
            change_label=args.change_label
        )
        
        if not json_data:
            print("No video data extracted from dataset")
            return 1
        
        print(f"\nExtracted {len(json_data)} video entries")
        
        # Show some examples
        print("\nFirst few entries:")
        for i, entry in enumerate(json_data[:3]):
            print(f"  {i+1}. Video: {entry['video']}")
            print(f"     Label: {entry['label']}")
        
        # Write to output file
        with open(args.output, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        print(f"\nJSON file created: {args.output}")
        print(f"Contains {len(json_data)} video entries")
        
        # Show label distribution
        labels = [item['label'] for item in json_data]
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            print(f"\nLabel distribution:")
            for label in sorted(unique_labels):
                count = labels.count(label)
                print(f"  {label}: {count} videos")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    main() 