"""# VQAScore Setup & Usage Guide

This guide explains how to set up the environment, sync files, generate input data, and run the VQAScore pipeline using [t2v_metrics](https://github.com/chancharikmitra/t2v_metrics) on the Autobot cluster.

---

## 📦 Environment Setup

1. **Create and activate the Conda environment (Python 3.10):**

```bash
conda create -n t2v python=3.10 -y
conda activate t2v
conda install pip -y
```

2. **Install required packages:**

```bash
conda install ffmpeg -c conda-forge
pip install git+https://github.com/chancharikmitra/t2v_metrics.git
```

---

## 🔐 Connect to Autobot Cluster

```bash
ssh autobot-0-37
nvidia-smi   # Confirm GPU availability
```

To view cluster status via web tunnel:

```bash
ssh -i /Users/jackieli/.ssh/jackie_autobot \\
    -L 127.0.0.1:53240:localhost:53240 \\
    jiayaoli@autobot.vision.cs.cmu.edu
```

---

## 📂 File Syncing with `rsync`

### Upload the full repository:

```bash
rsync -avz /Users/jackieli/Downloads/t2v_metrics autobot:/project_data/ramanan/jiayaoli/
```

### Delete extraneous remote files:

```bash
rsync -avz --delete /local/path autobot:/remote/path
```

### Sync `input.json`:

```bash
rsync /Users/jackieli/Downloads/prof_code/vqa_pre_post/input.json autobot:/project_data/ramanan/jiayaoli/vqascore/
```

### Retrieve results:

```bash
scp autobot:/project_data/ramanan/jiayaoli/vqascore/input_scored.json /Users/jackieli/Downloads/prof_code/vqa_pre_post/
```

---

## 🛠️ Create `input.json`

### From a local directory of videos:

```bash
python video_dir_to_input.py data/
```

### From Hugging Face DB with a specific label:

```bash
python hf_db_to_input.py \\
    -database jackieyayqli/vqascore \\
    -label cam_motion.dolly_zoom_movement.has_dolly_in_zoom_out
```

### Modify labels in existing input:

```bash
python hf_db_to_input.py --change_label "cam_motion.camera_centric_movement.roll_counterclockwise.only_roll_counterclockwise"
```

---

## 🚀 Run VQAScore

### Single-threaded scoring:

```bash
python score.py -r all_labels.json -i input.json
```

### Parallel scoring on 8 GPUs:

```bash
bash run_parallel_eval.sh input.json all_labels.json 8
```

### Specify GPU IDs manually:

```bash
bash run_parallel_eval.sh input2.json all_labels.json "0,1,2,3,4,5,6,7"
```

---

## 📁 Directory Structure

Recommended file layout on Autobot:

```
/project_data/ramanan/jiayaoli/
└── vqascore/
    ├── input.json
    ├── input_scored.json
    ├── all_labels.json
    └── ...
```

---

## 💡 Notes

- Be cautious with the `--delete` flag in `rsync`; it will remove files from the remote directory that don’t exist locally.
- Always run `nvidia-smi` after connecting to verify GPU availability.
- `all_labels.json` contains available label definitions.

---

For any issues or contributions, refer to the original [t2v_metrics GitHub repository](https://github.com/chancharikmitra/t2v_metrics).
"""