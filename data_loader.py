#!/usr/bin/env python3
"""
Data loading utilities for video action recognition
Handles video preprocessing, frame extraction, and data augmentation
"""

import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch
from torchvision import transforms
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class VideoDataset(Dataset):
    """Custom dataset for video action recognition"""
    
    def __init__(self, csv_file, video_dir, num_frames=16, frame_size=(112, 112), 
                 transform=None, is_training=True):
        """
        Args:
            csv_file (str): Path to CSV file with video labels
            video_dir (str): Directory containing video files
            num_frames (int): Number of frames to extract from each video
            frame_size (tuple): Target frame size (height, width)
            transform (callable): Optional transform to be applied on frames
            is_training (bool): Whether this is training data (affects augmentation)
        """
        self.data = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        self.is_training = is_training
        
        # Create label encoder if not provided
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data['label'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.data.iloc[idx]['video_name'])
        label = self.labels[idx]
        
        # Extract frames from video
        frames = self.extract_frames(video_path)
        
        # Apply transforms if provided
        if self.transform:
            frames = self.transform(frames)
        
        return frames, label
    
    def extract_frames(self, video_path):
        """Extract frames from video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"No frames in video: {video_path}")
        
        # Sample frames evenly across the video
        if self.is_training:
            # Random sampling for training
            frame_indices = sorted(random.sample(range(total_frames), 
                                               min(self.num_frames, total_frames)))
        else:
            # Uniform sampling for validation/test
            frame_indices = np.linspace(0, total_frames - 1, 
                                      min(self.num_frames, total_frames), dtype=int)
        
        frames = []
        frame_idx = 0
        
        while cap.isOpened() and len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in frame_indices:
                # Resize and convert to RGB
                frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        
        # Pad with last frame if we don't have enough frames
        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8))
        
        return np.array(frames[:self.num_frames])

class VideoTransform:
    """Custom transform for video data"""
    
    def __init__(self, is_training=True):
        self.is_training = is_training
        
    def __call__(self, frames):
        """
        Args:
            frames: numpy array of shape (T, H, W, C)
        Returns:
            torch tensor of shape (T, C, H, W)
        """
        # Convert to tensor and normalize
        frames_tensor = []
        for frame in frames:
            frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
            frames_tensor.append(frame_tensor)
        
        # Stack frames
        frames_tensor = torch.stack(frames_tensor)  # Shape: (T, C, H, W)
        
        # Apply augmentation if training
        if self.is_training:
            frames_tensor = self.augment_frames(frames_tensor)
        
        # Kinetics-400 normalization
        kinetics_mean = [0.43216, 0.394666, 0.37645]
        kinetics_std = [0.22803, 0.22145, 0.216989]
        
        # Normalize
        for i in range(3):
            frames_tensor[:, i, :, :] = (frames_tensor[:, i, :, :] - kinetics_mean[i]) / kinetics_std[i]
        
        return frames_tensor
    
    def augment_frames(self, frames):
        """Apply data augmentation to video frames"""
        # Random horizontal flip
        if random.random() > 0.5:
            frames = torch.flip(frames, dims=[3])
        
        # Random brightness adjustment
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            frames = frames * brightness_factor
            frames = torch.clamp(frames, 0, 1)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = frames.mean()
            frames = (frames - mean) * contrast_factor + mean
            frames = torch.clamp(frames, 0, 1)
        
        return frames

def create_data_loaders(train_csv, val_csv, test_csv, video_dir, 
                       batch_size=32, num_frames=16, frame_size=(112, 112),
                       num_workers=4):
    """Create data loaders for training, validation, and testing"""
    
    # Create transforms
    train_transform = VideoTransform(is_training=True)
    val_transform = VideoTransform(is_training=False)
    
    # Create datasets
    train_dataset = VideoDataset(
        train_csv, video_dir, num_frames, frame_size, 
        train_transform, is_training=True
    )
    
    val_dataset = VideoDataset(
        val_csv, video_dir, num_frames, frame_size,
        val_transform, is_training=False
    )
    
    test_dataset = VideoDataset(
        test_csv, video_dir, num_frames, frame_size,
        val_transform, is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.label_encoder

def save_label_encoder(label_encoder, filepath):
    """Save label encoder to file"""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(label_encoder, f)

def load_label_encoder(filepath):
    """Load label encoder from file"""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Test the data loader
    print("Testing VideoDataset...")
    
    # Example usage
    train_csv = "archive/train.csv"
    val_csv = "archive/val.csv"
    test_csv = "archive/test.csv"
    video_dir = "archive/train"
    
    if os.path.exists(train_csv):
        train_loader, val_loader, test_loader, label_encoder = create_data_loaders(
            train_csv, val_csv, test_csv, video_dir, batch_size=4
        )
        
        print(f"Number of classes: {len(label_encoder.classes_)}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test a batch
        for frames, labels in train_loader:
            print(f"Batch shape: {frames.shape}")
            print(f"Labels: {labels}")
            break
    else:
        print("CSV files not found. Please check the paths.")
