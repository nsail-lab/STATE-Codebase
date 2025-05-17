# STATE-Codebase

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- Pillow
- plotly (for visualization)

## Usage

### Training

Train the STATE model with the following command:

```bash
python STATE.py train \
  --data_dir /path/to/trajectory/dataset \
  --annotations_file /path/to/annotations.json \
  --output_dir state_output \
  --threshold 3 \
  --batch_size 64 \
  --epochs 2000 \
  --threat_model_path /path/to/classifier.pth \
  --cuda
```

### Inference

Generate trajectory images with a trained model:

```bash
python STATE.py inference \
  --generator_path /path/to/trained/generator.pth \
  --data_dir /path/to/trajectory/dataset \
  --annotations_file /path/to/annotations.json \
  --output_dir inference_output \
  --num_samples 10 \
  --images_per_sample 5 \
  --cuda
```

### Visualization

Use the example notebook `example_plot_for_paper.ipynb` for visualization of generated trajectories.

## Dataset Format

The training dataset should have the following structure:
```bash
data_dir/
├── trajectory_id_1/
│   ├── trajectory_id_1_trajectory.png  # Binary trajectory image
│   └── trajectory_id_1_CLIP.npy        # CLIP feature array (5, 512)
├── trajectory_id_2/
│   ├── trajectory_id_2_trajectory.png
│   └── trajectory_id_2_CLIP.npy
└── ...
```
The annotations file should be a JSON file containing threat scores for each trajectory.