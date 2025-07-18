#!/usr/bin/env python3
"""
Script to convert a directory of videos into a JSON input file.
Takes a directory path and generates a JSON file with video paths and labels.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

# Common video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}

def find_video_files(directory: str) -> List[str]:
    """
    Find all video files in the given directory.
    
    Args:
        directory: Path to the directory containing videos
        
    Returns:
        List of video file paths
    """
    video_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    
    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
            video_files.append(str(file_path))
    
    return sorted(video_files)

def create_json_input(video_files: List[str], label: str = "default_label") -> List[Dict[str, str]]:
    """
    Create JSON input structure from video files.
    
    Args:
        video_files: List of video file paths
        label: Label to assign to all videos
        
    Returns:
        List of dictionaries with video and label keys
    """
    return [{"video": video_path, "label": label} for video_path in video_files]

def main():
    parser = argparse.ArgumentParser(description="Convert directory of videos to JSON input file")
    parser.add_argument("video_dir", help="Directory containing video files")
    parser.add_argument("-o", "--output", default="input.json", help="Output JSON file (default: input.json)")
    parser.add_argument("-l", "--label", default="cam_motion.steadiness_and_movement.moving_camera", 
                       help="Label to assign to all videos (default: cam_motion.steadiness_and_movement.moving_camera)")
    parser.add_argument("--relative-paths", action="store_true", 
                       help="Use relative paths instead of absolute paths")
    
    args = parser.parse_args()
    
    try:
        # Find video files
        video_files = find_video_files(args.video_dir)
        
        if not video_files:
            print(f"No video files found in {args.video_dir}")
            return
        
        print(f"Found {len(video_files)} video files:")
        for video in video_files:
            print(f"  - {video}")
        
        # Convert to relative paths if requested
        if args.relative_paths:
            video_files = [os.path.relpath(video) for video in video_files]
        
        # Create JSON structure
        json_data = create_json_input(video_files, args.label)
        
        # Write to output file
        with open(args.output, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        print(f"\nJSON file created: {args.output}")
        print(f"Contains {len(json_data)} video entries with label: '{args.label}'")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    main()
