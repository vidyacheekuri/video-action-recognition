#!/usr/bin/env python3
"""
Utility functions for video action recognition
Includes training utilities, metrics, and helper functions
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save the best model weights"""
        self.best_weights = model.state_dict().copy()

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), data.size(0))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                  f'Loss: {losses.avg:.4f}, Acc@1: {top1.avg:.4f}, Acc@5: {top5.avg:.4f}')
    
    return losses.avg, top1.avg, top5.avg

def validate_epoch(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            # Update metrics
            losses.update(loss.item(), data.size(0))
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))
    
    return losses.avg, top1.avg, top5.avg

def save_checkpoint(model, optimizer, epoch, loss, acc, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    acc = checkpoint['accuracy']
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    return epoch, loss, acc

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def evaluate_model(model, test_loader, device, class_names):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    test_loss, test_acc1, test_acc5 = validate_epoch(
        model, test_loader, nn.CrossEntropyLoss(), device
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy@1: {test_acc1:.4f}")
    print(f"Test Accuracy@5: {test_acc5:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    return all_targets, all_preds, test_acc1, test_acc5

def get_model_summary(model, input_size=(1, 16, 3, 112, 112)):
    """Get model summary"""
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")

def save_training_config(config, filepath):
    """Save training configuration to JSON"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training configuration saved to {filepath}")

def load_training_config(filepath):
    """Load training configuration from JSON"""
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"Training configuration loaded from {filepath}")
    return config

def create_experiment_directory(experiment_name):
    """Create directory for experiment results"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/{experiment_name}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_dir}/plots", exist_ok=True)
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir

if __name__ == "__main__":
    print("Utility functions for video action recognition")
    print("Available functions:")
    print("- EarlyStopping: Early stopping utility")
    print("- AverageMeter: Metric tracking")
    print("- accuracy: Top-k accuracy calculation")
    print("- train_epoch: Training loop")
    print("- validate_epoch: Validation loop")
    print("- save_checkpoint: Save model checkpoint")
    print("- load_checkpoint: Load model checkpoint")
    print("- plot_training_history: Plot training curves")
    print("- plot_confusion_matrix: Plot confusion matrix")
    print("- evaluate_model: Comprehensive evaluation")
    print("- get_model_summary: Model summary")
