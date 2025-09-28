#!/usr/bin/env python3
"""
Single video prediction script for web interface
Takes a video file and outputs prediction results as JSON
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import pickle
import json
import argparse
from model import R3D18Classifier, CNNLSTM
import warnings
warnings.filterwarnings('ignore')

# Configuration
VIDEO_SIZE = (112, 112)
NUM_FRAMES = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes, device):
    """Load a trained model from .pth file"""
    try:
        # Try R3D-18 first
        model = R3D18Classifier(num_classes=num_classes, pretrained=False, freeze_backbone=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        try:
            # Fallback to CNNLSTM
            model = CNNLSTM(num_classes=num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model
        except Exception as e2:
            raise Exception(f"Failed to load model: {e2}")

def load_video_frames(video_path, num_frames=16, target_size=(112, 112)):
    """Extract frames from video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"No frames in video: {video_path}")
    
    # Sample frames evenly across the video
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    frame_idx = 0
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in frame_indices:
            # Resize and convert to RGB
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    
    # Pad with last frame if we don't have enough frames
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1])
        else:
            frames.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
    
    return frames[:num_frames]

def preprocess_frames(frames):
    """Preprocess frames for model input"""
    # Convert to tensor format
    frames_tensor = []
    for frame in frames:
        # Convert to tensor and normalize
        frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
        frames_tensor.append(frame_tensor)
    
    # Stack frames and add batch dimension
    frames_tensor = torch.stack(frames_tensor).unsqueeze(0)  # Shape: (1, T, C, H, W)
    
    # Kinetics-400 normalization
    kinetics_mean = [0.43216, 0.394666, 0.37645]
    kinetics_std = [0.22803, 0.22145, 0.216989]
    
    # Normalize
    for i in range(3):
        frames_tensor[:, :, i, :, :] = (frames_tensor[:, :, i, :, :] - kinetics_mean[i]) / kinetics_std[i]
    
    return frames_tensor

def predict_video(model, video_path, label_encoder, device):
    """Predict action in a video"""
    # Load and preprocess video
    frames = load_video_frames(video_path, NUM_FRAMES, VIDEO_SIZE)
    input_tensor = preprocess_frames(frames).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()
    
    # Get class name
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    
    # Get top 5 predictions
    all_probs = probabilities[0].cpu().numpy()
    top5_indices = np.argsort(all_probs)[-5:][::-1]
    
    top_predictions = []
    for idx in top5_indices:
        class_name = label_encoder.inverse_transform([idx])[0]
        prob = all_probs[idx]
        top_predictions.append({
            "action": class_name,
            "score": float(prob)
        })
    
    return {
        "predictedAction": predicted_class,
        "confidence": float(confidence),
        "topPredictions": top_predictions
    }

def main():
    parser = argparse.ArgumentParser(description='Predict action in a single video')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--model', default='best_model.pth', help='Model file path')
    
    args = parser.parse_args()
    
    try:
        # Load label encoder
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        
        # Load model
        num_classes = len(label_encoder.classes_)
        model = load_model(args.model, num_classes, DEVICE)
        
        # Predict video
        result = predict_video(model, args.video, label_encoder, DEVICE)
        
        # Save results to JSON file
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Prediction completed. Results saved to {args.output}")
        
    except Exception as e:
        error_result = {"error": str(e)}
        with open(args.output, 'w') as f:
            json.dump(error_result, f, indent=2)
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
