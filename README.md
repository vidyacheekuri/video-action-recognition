# Video Action Recognition

A deep learning project for video action recognition using PyTorch. This project implements both CNN+LSTM and R3D-18 architectures to classify human actions in video clips.

## 🎯 Features

- **Multiple Model Architectures**: CNN+LSTM and R3D-18 (3D ResNet)
- **Real-time Prediction**: Web interface for instant video analysis
- **Modern UI**: Clean, responsive web interface with drag-and-drop upload
- **Comprehensive Training**: Full training pipeline with data augmentation
- **Model Evaluation**: Detailed metrics and visualization tools

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd video-action-recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download UCF-101 dataset and extract to `archive/` directory
   - Or use your own video dataset with CSV labels

4. **Run the web application**
   ```bash
   python app.py
   ```
   Open your browser and go to `http://localhost:5001`

## 📁 Project Structure

```
video-action-recognition/
├── app.py                      # Flask web server
├── video_predictor.html        # Web interface
├── model.py                    # Model architectures
├── predict_single_video.py     # Single video prediction
├── train.py                    # Training script
├── data_loader.py              # Data loading utilities
├── utils.py                    # Training utilities
├── requirements.txt            # Python dependencies
├── best_model.pth              # Trained model weights
├── label_encoder.pkl           # Label encoding
├── archive/                    # Dataset directory
│   ├── train/                  # Training videos
│   ├── val/                    # Validation videos
│   ├── test/                   # Test videos
│   ├── train.csv               # Training labels
│   ├── val.csv                 # Validation labels
│   └── test.csv                # Test labels
└── experiments/                # Training outputs
    └── experiment_name_timestamp/
        ├── checkpoints/         # Model checkpoints
        ├── plots/              # Training plots
        ├── config.json         # Training config
        └── results.json        # Training results
```

## 🏋️ Training

### Basic Training

Train a model with default settings:

```bash
python train.py --model r3d18 --epochs 50
```

### Advanced Training

```bash
python train.py \
    --model cnnlstm \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --num_frames 32 \
    --frame_size 224 224 \
    --experiment_name my_experiment
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `r3d18` | Model architecture (`r3d18`, `cnnlstm`) |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `32` | Batch size for training |
| `--lr` | `0.001` | Learning rate |
| `--num_frames` | `16` | Number of frames per video |
| `--frame_size` | `112 112` | Frame dimensions (height width) |
| `--pretrained` | `False` | Use pretrained backbone |
| `--freeze_backbone` | `False` | Freeze backbone weights |
| `--patience` | `10` | Early stopping patience |
| `--scheduler` | `plateau` | LR scheduler (`plateau`, `cosine`) |

### Resume Training

```bash
python train.py --resume experiments/my_experiment/checkpoints/checkpoint_epoch_50.pth
```

## 🔮 Prediction

### Web Interface

1. Start the web server:
   ```bash
   python app.py
   ```

2. Open `http://localhost:5001` in your browser

3. Upload a video file and click "Submit" to get predictions

### Command Line

```bash
python predict_single_video.py --video path/to/video.mp4 --output results.json
```

## 📊 Model Architectures

### R3D-18 (3D ResNet-18)
- **Architecture**: 3D convolutional neural network
- **Input**: Video clips (T, C, H, W)
- **Advantages**: Good for spatial-temporal features
- **Use Case**: General video action recognition

### CNN+LSTM
- **Architecture**: 2D CNN + LSTM
- **Input**: Sequence of frames
- **Advantages**: Explicit temporal modeling
- **Use Case**: When temporal relationships are crucial

## 📈 Performance

| Model | Top-1 Accuracy | Top-5 Accuracy | Parameters |
|-------|----------------|----------------|------------|
| R3D-18 | ~85% | ~95% | ~33M |
| CNN+LSTM | ~82% | ~93% | ~15M |

*Results on UCF-101 dataset (may vary based on training configuration)*

## 🛠️ Development

### Adding New Models

1. Define your model in `model.py`:
   ```python
   class MyModel(nn.Module):
       def __init__(self, num_classes):
           super().__init__()
           # Your model definition
       
       def forward(self, x):
           # Your forward pass
           return output
   ```

2. Add model option to `train.py`:
   ```python
   elif model_name == 'mymodel':
       model = MyModel(num_classes=num_classes)
   ```

### Custom Dataset

1. Prepare your dataset in the following format:
   ```
   dataset/
   ├── videos/
   │   ├── video1.mp4
   │   ├── video2.mp4
   │   └── ...
   ├── train.csv
   ├── val.csv
   └── test.csv
   ```

2. CSV format:
   ```csv
   video_name,label
   video1.mp4,action1
   video2.mp4,action2
   ```

3. Update data paths in `train.py` and `data_loader.py`

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for dataset

### Software
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- FFmpeg (for video processing)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 8`
   - Reduce frame size: `--frame_size 64 64`
   - Use gradient accumulation

2. **Slow Training**
   - Increase number of workers: `--num_workers 8`
   - Use mixed precision training
   - Ensure data is on SSD

3. **Poor Accuracy**
   - Increase number of frames: `--num_frames 32`
   - Use data augmentation
   - Try different learning rates

### Getting Help

- Check the logs in `experiments/` directory
- Verify your dataset format
- Ensure all dependencies are installed correctly

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 References

- [UCF-101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
- [R3D: A Purely Convolutional Network for Video Classification](https://arxiv.org/abs/1711.11248)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🙏 Acknowledgments

- UCF-101 dataset creators
- PyTorch team
- OpenCV community
- Flask framework

---

**Happy Training! 🎬🤖**
