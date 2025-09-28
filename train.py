#!/usr/bin/env python3
"""
Training script for video action recognition
Trains CNN+LSTM and R3D-18 models on UCF-101 dataset
"""

import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from model import R3D18Classifier, CNNLSTM
from data_loader import create_data_loaders, save_label_encoder
from utils import (EarlyStopping, AverageMeter, train_epoch, validate_epoch,
                   save_checkpoint, load_checkpoint, plot_training_history,
                   evaluate_model, get_model_summary, save_training_config,
                   create_experiment_directory)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train video action recognition model')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='r3d18', 
                       choices=['r3d18', 'cnnlstm'],
                       help='Model architecture to train')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained backbone')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone weights')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='archive',
                       help='Directory containing dataset')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames per video')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[112, 112],
                       help='Frame size (height, width)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine'],
                       help='Learning rate scheduler')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, default='video_action_recognition',
                       help='Name of the experiment')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup training device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def create_model(model_name, num_classes, pretrained=False, freeze_backbone=False):
    """Create model based on architecture name"""
    if model_name == 'r3d18':
        model = R3D18Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    elif model_name == 'cnnlstm':
        model = CNNLSTM(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def train_model(model, train_loader, val_loader, test_loader, args, device, exp_dir):
    """Main training loop"""
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Setup loss and early stopping
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1
    
    print(f"Starting training from epoch {start_epoch}")
    print(f"Total epochs: {args.epochs}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.scheduler == 'cosine':
            scheduler.step()
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc1.item())
        val_accs.append(val_acc1.item())
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc@1: {train_acc1:.4f}, Train Acc@5: {train_acc5:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc@1: {val_acc1:.4f}, Val Acc@5: {val_acc5:.4f}")
        
        # Save best model
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc1,
                os.path.join(exp_dir, 'checkpoints', 'best_model.pth')
            )
            print(f"New best model saved! Val Acc: {val_acc1:.4f}")
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc1,
                os.path.join(exp_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Plot training history
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(exp_dir, 'plots', 'training_history.png')
    )
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    # Load best model for evaluation
    best_checkpoint = os.path.join(exp_dir, 'checkpoints', 'best_model.pth')
    if os.path.exists(best_checkpoint):
        load_checkpoint(best_checkpoint, model)
        print("Loaded best model for evaluation")
    
    # Get class names
    class_names = train_loader.dataset.label_encoder.classes_
    
    # Evaluate on test set
    test_targets, test_preds, test_acc1, test_acc5 = evaluate_model(
        model, test_loader, device, class_names
    )
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(exp_dir, 'final_model.pth'))
    print(f"Final model saved to {os.path.join(exp_dir, 'final_model.pth')}")
    
    # Save training results
    results = {
        'best_val_acc': best_val_acc.item(),
        'final_test_acc1': test_acc1.item(),
        'final_test_acc5': test_acc5.item(),
        'total_epochs': epoch + 1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy: {test_acc1:.4f}")
    print(f"Results saved to: {exp_dir}")

def main():
    """Main function"""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Create experiment directory
    exp_dir = create_experiment_directory(args.experiment_name)
    
    # Save training configuration
    config = vars(args)
    config['device'] = str(device)
    save_training_config(config, os.path.join(exp_dir, 'config.json'))
    
    # Setup data paths
    train_csv = os.path.join(args.data_dir, 'train.csv')
    val_csv = os.path.join(args.data_dir, 'val.csv')
    test_csv = os.path.join(args.data_dir, 'test.csv')
    video_dir = os.path.join(args.data_dir, 'train')
    
    # Check if data exists
    if not all(os.path.exists(f) for f in [train_csv, val_csv, test_csv]):
        print("Error: Dataset files not found!")
        print(f"Expected files: {train_csv}, {val_csv}, {test_csv}")
        return
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, label_encoder = create_data_loaders(
        train_csv, val_csv, test_csv, video_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_size=tuple(args.frame_size),
        num_workers=args.num_workers
    )
    
    # Save label encoder
    save_label_encoder(label_encoder, os.path.join(exp_dir, 'label_encoder.pkl'))
    print(f"Label encoder saved to {os.path.join(exp_dir, 'label_encoder.pkl')}")
    
    # Create model
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    
    model = create_model(
        args.model, num_classes, 
        pretrained=args.pretrained, 
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    # Print model summary
    get_model_summary(model, input_size=(1, args.num_frames, 3, args.frame_size[0], args.frame_size[1]))
    
    # Train model
    train_model(model, train_loader, val_loader, test_loader, args, device, exp_dir)

if __name__ == "__main__":
    main()
