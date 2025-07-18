import json
import os
import argparse
import t2v_metrics
import torch
from tqdm import tqdm

def evaluate(input_file, ref_file):
    # Load video-label pairs
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Load reference prompt templates
    with open(ref_file, 'r') as f:
        ref_prompts = json.load(f)

    # Initialize scorer
    gem_score = t2v_metrics.VQAScore(model='qwen2.5-vl-7b-cambench')

    output_data = []
    for entry in tqdm(data):
        video_path = entry['video']
        label_key = entry['label']

        # Get question template from ref file
        if label_key not in ref_prompts:
            print(f"[Warning] Label '{label_key}' not found in reference file. Skipping.")
            continue

        question_template = ref_prompts[label_key]["definition"] + "Please only answer Yes or No."

        try:
            score_tensor = gem_score(images=[video_path], texts=[label_key], question_template=question_template)
            score = score_tensor.item() if isinstance(score_tensor, torch.Tensor) else float(score_tensor)
        except Exception as e:
            print(f"[Error] Failed to score video: {video_path} for label: {label_key}. Error: {e}")
            score = None

        output_data.append({
            "video": video_path,
            "label": label_key,
            "question": question_template,
            "score": score
        })

    # Save results
    output_file = os.path.splitext(input_file)[0] + "_scored.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"[Done] Results written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to input JSON with 'video' and 'label' keys")
    parser.add_argument("-r", "--ref_file", type=str, required=True, help="Path to reference JSON with label -> question_template mapping")
    args = parser.parse_args()

    evaluate(args.input_file, args.ref_file)
