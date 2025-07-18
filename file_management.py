import json
import math
import os
import argparse


def split_data(input_file, num_gpus, output_dir):
    """Split input JSON file into chunks for each GPU"""
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)

    total_items = len(data)
    chunk_size = math.ceil(total_items / num_gpus)

    print(f'[Info] Total items: {total_items}')
    print(f'[Info] Chunk size: {chunk_size}')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create chunks
    chunk_files = []
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * chunk_size
        end_idx = min((gpu_id + 1) * chunk_size, total_items)
        chunk = data[start_idx:end_idx]
        
        if chunk:  # Only create non-empty chunks
            chunk_file = os.path.join(output_dir, f'chunk_{gpu_id}.json')
            with open(chunk_file, 'w') as f:
                json.dump(chunk, f, indent=2)
            chunk_files.append(chunk_file)
            print(f'[Info] GPU {gpu_id}: {len(chunk)} items -> chunk_{gpu_id}.json')
        else:
            print(f'[Info] GPU {gpu_id}: 0 items (skipped)')
    
    return chunk_files


def merge_results(output_files, final_output):
    """Merge results from multiple GPU output files"""
    all_results = []
    total_processed = 0

    for output_file in output_files:
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    results = json.load(f)
                all_results.extend(results)
                total_processed += len(results)
                print(f'[Info] Merged {len(results)} results from {os.path.basename(output_file)}')
            except Exception as e:
                print(f'[Warning] Failed to load {output_file}: {e}')
        else:
            print(f'[Warning] Output file not found: {output_file}')

    # Write merged results
    with open(final_output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'[Info] Total merged results: {total_processed}')
    return total_processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["split", "merge"], help="Command to execute")
    parser.add_argument("--input_file", help="Input JSON file (for split command)")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs (for split command)")
    parser.add_argument("--output_dir", help="Output directory (for split command)")
    parser.add_argument("--output_files", nargs="+", help="List of output files to merge (for merge command)")
    parser.add_argument("--final_output", help="Final merged output file (for merge command)")
    
    args = parser.parse_args()
    
    if args.command == "split":
        if not all([args.input_file, args.num_gpus, args.output_dir]):
            print("Error: split command requires --input_file, --num_gpus, and --output_dir")
            exit(1)
        split_data(args.input_file, args.num_gpus, args.output_dir)
    
    elif args.command == "merge":
        if not all([args.output_files, args.final_output]):
            print("Error: merge command requires --output_files and --final_output")
            exit(1)
        merge_results(args.output_files, args.final_output)