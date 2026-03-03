"""
Training Script for SAM with LoRA and LRU

This script trains a SAM model with:
1. LoRA (Low-Rank Adaptation) for efficient fine-tuning
2. LRU (Linear Recurrent Unit) for temporal sequence modeling

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_SAM_LRU.py --config config.yaml --experiment-id 1

For resuming training:
    CUDA_VISIBLE_DEVICES=0 python train_SAM_LRU.py --config config.yaml \
        --resume --checkpoint path/to/checkpoint.pth --experiment-id 2
"""

import numpy as np
import torch
import monai
import yaml
import os
import argparse
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F
from medpy import metric
import sys

sys.path.append('/home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA')

from src.dataloader_sequential import DatasetSegmentationSequential, collate_fn_sequential
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b
import src.utils as utils
from SAM_LRU import LRU_SAM


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SAM-LoRA-LRU Training')
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to checkpoint for resuming')
    parser.add_argument('--experiment-id', type=int, default=1,
                        help='Experiment ID for logging')
    parser.add_argument('--lru-hidden-dim', type=int, default=512,
                        help='Hidden dimension for LRU')
    parser.add_argument('--disable-lru', action='store_true',
                        help='Disable LRU (baseline LoRA only)')
    return parser.parse_args()


def calculate_metrics(pred, target):
    """
    Calculate segmentation metrics.

    Arguments:
        pred: Predicted mask (tensor)
        target: Ground truth mask (tensor)

    Returns:
        dice: Dice score
        iou: IoU score
        hd95: Hausdorff distance 95
    """
    pred_binary = (pred > 0.5).float()
    label_binary = (target > 0.5).float()
    pred_binary = pred_binary.cpu().numpy().astype(bool)
    label_binary = label_binary.cpu().numpy().astype(bool)

    intersection = np.logical_and(pred_binary, label_binary)
    union = np.logical_or(pred_binary, label_binary)
    dice = (2.0 * np.sum(intersection)) / (np.sum(pred_binary) + np.sum(label_binary) + 1e-8)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

    try:
        if np.sum(pred_binary) > 0 and np.sum(label_binary) > 0:
            hd95 = metric.binary.hd95(pred_binary, label_binary)
        else:
            hd95 = np.nan
    except:
        hd95 = np.nan

    return dice, iou, hd95


def calculate_iou(pred, target):
    """Calculate IoU score only (faster than full metrics)."""
    pred_binary = (pred > 0.5).float()
    label_binary = (target > 0.5).float()
    pred_binary = pred_binary.cpu().numpy().astype(bool)
    label_binary = label_binary.cpu().numpy().astype(bool)

    intersection = np.logical_and(pred_binary, label_binary)
    union = np.logical_or(pred_binary, label_binary)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

    return iou


def train_epoch(model, dataloader, optimizer, seg_loss, device, epoch_idx, num_epochs):
    """
    Train for one epoch.

    Arguments:
        model: LRU_SAM model
        dataloader: Training dataloader (sequential)
        optimizer: Optimizer
        seg_loss: Loss function
        device: Device (cuda/cpu)
        epoch_idx: Current epoch number
        num_epochs: Total number of epochs

    Returns:
        epoch_train_loss: Average training loss
        epoch_train_iou: Average training IoU
    """
    model.sam.train()  # Set SAM to train mode
    if model.lru is not None:
        model.lru.train()  # Set LRU to train mode

    epoch_losses = []
    epoch_ious = []

    # Track current video ID to reset hidden state
    current_video_id = None

    for i, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch_idx}/{num_epochs}")):
        # Check if we're starting a new video sequence
        batch_video_id = batch[0].get("video_id", None)
        if batch_video_id is not None and batch_video_id != current_video_id:
            # Reset hidden state for new video
            model.reset_hidden_state()
            current_video_id = batch_video_id

        # Move batch to device
        device_batch = []
        for item in batch:
            device_item = {}
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    device_item[key] = value.to(device)
                else:
                    device_item[key] = value
            device_batch.append(device_item)

        # Forward pass
        outputs = model(
            batched_input=device_batch,
            multimask_output=False
        )

        # Calculate loss
        stk_gt, stk_out = utils.stacking_batch(device_batch, outputs)
        stk_out = stk_out.squeeze(1)
        stk_gt = stk_gt.unsqueeze(1)
        loss = seg_loss(stk_out, stk_gt.float().to(device))
        iou = calculate_iou(stk_out, stk_gt.float().to(device))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record metrics
        epoch_losses.append(loss.item())
        epoch_ious.append(iou)

    # Calculate epoch metrics
    epoch_train_loss = np.mean(epoch_losses)
    epoch_train_iou = np.mean(epoch_ious)

    return epoch_train_loss, epoch_train_iou


def validate(model, dataloader, seg_loss, device):
    """
    Validate the model.

    Arguments:
        model: LRU_SAM model
        dataloader: Validation dataloader (sequential)
        seg_loss: Loss function
        device: Device (cuda/cpu)

    Returns:
        val_metrics: Dictionary with validation metrics
    """
    model.sam.eval()
    if model.lru is not None:
        model.lru.eval()

    val_losses = []
    val_dices = []
    val_ious = []
    val_hd95s = []

    # Track current video ID
    current_video_id = None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Check if starting new video
            batch_video_id = batch[0].get("video_id", None)
            if batch_video_id is not None and batch_video_id != current_video_id:
                model.reset_hidden_state()
                current_video_id = batch_video_id

            # Move batch to device
            device_batch = []
            for item in batch:
                device_item = {}
                for key, value in item.items():
                    if isinstance(value, torch.Tensor):
                        device_item[key] = value.to(device)
                    else:
                        device_item[key] = value
                device_batch.append(device_item)

            # Forward pass
            outputs = model(
                batched_input=device_batch,
                multimask_output=False
            )

            # Calculate metrics
            stk_gt, stk_out = utils.stacking_batch(device_batch, outputs)
            stk_out = stk_out.squeeze(1)
            stk_gt = stk_gt.unsqueeze(1)

            val_loss = seg_loss(stk_out, stk_gt.float().to(device))
            dice, iou, hd95 = calculate_metrics(stk_out, stk_gt.float().to(device))

            val_losses.append(val_loss.item())
            val_dices.append(dice)
            val_ious.append(iou)
            val_hd95s.append(hd95)

    # Calculate mean metrics
    val_metrics = {
        'loss': np.mean(val_losses),
        'dice': np.mean(val_dices),
        'iou': np.mean(val_ious),
        'hd95': np.mean([x for x in val_hd95s if not np.isnan(x)]) if any(
            not np.isnan(x) for x in val_hd95s) else np.nan,
        'valid_hd95_count': len([x for x in val_hd95s if not np.isnan(x)])
    }

    return val_metrics


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as ymlfile:
        config_file = yaml.safe_load(ymlfile)

    # Create experiment directory
    exp_dir = os.path.join("/mnt/hdd2/task2/sam_lora_lru", f"exp_{args.experiment_id}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize base SAM model
    sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])

    # Initialize LRU_SAM model
    model = LRU_SAM(
        sam_model=sam,
        lora_rank=config_file["SAM"]["RANK"],
        lru_hidden_dim=args.lru_hidden_dim,
        use_lru=not args.disable_lru
    )

    # Create processor
    processor = Samprocessor(model.sam)

    # Create datasets with sequential loading
    print("Loading training dataset...")
    train_ds = DatasetSegmentationSequential(config_file, processor, mode="train2")
    train_dataloader = DataLoader(
        train_ds,
        batch_size=config_file["TRAIN"]["BATCH_SIZE"],
        shuffle=False,  # IMPORTANT: No shuffle for sequential data
        collate_fn=collate_fn_sequential
    )

    print("Loading validation dataset...")
    val_ds = DatasetSegmentationSequential(config_file, processor, mode="val2")
    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,  # Batch size 1 for validation
        shuffle=False,
        collate_fn=collate_fn_sequential
    )

    # Print video sequence info
    print("\nTraining dataset video sequences:")
    train_videos = train_ds.get_video_sequences()
    print(f"  Number of videos: {len(train_videos)}")
    print(f"  Frames per video: {[len(v) for v in list(train_videos.values())[:5]]}...")

    print("\nValidation dataset video sequences:")
    val_videos = val_ds.get_video_sequences()
    print(f"  Number of videos: {len(val_videos)}")
    print(f"  Frames per video: {[len(v) for v in list(val_videos.values())[:5]]}...")

    # Initialize optimizer (only for trainable parameters)
    trainable_params = model.get_trainable_parameters()
    optimizer = Adam(
        trainable_params,
        lr=config_file["TRAIN"]["LR"],
        weight_decay=config_file["TRAIN"].get("WEIGHT_DECAY", 0.0)
    )

    print(f"\nTrainable parameters:")
    print(f"  LoRA: {len(model.get_lora_parameters())} parameter groups")
    print(f"  LRU: {len(model.get_lru_parameters())} parameters" if model.use_lru else "  LRU: disabled")

    # Loss function
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # Training parameters
    num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]
    patience = config_file["TRAIN"].get("PATIENCE", 10)
    checkpoint_freq = config_file["TRAIN"].get("CHECKPOINT_FREQ", 5)

    # Move model to device
    model.sam.to(device)
    if model.lru is not None:
        model.lru.to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_iou = 0.0
    no_improve_epochs = 0
    train_loss_history = []
    train_iou_history = []
    val_loss_history = []
    val_dice_history = []
    val_iou_history = []
    val_hd95_history = []

    if args.resume and args.checkpoint:
        print(f"\nResuming from checkpoint: {args.checkpoint}")
        model.load_checkpoint(args.checkpoint)

        # Try to load training history
        history_path = os.path.join(exp_dir, "training_history.npz")
        if os.path.exists(history_path):
            history = np.load(history_path)
            train_loss_history = history["train_loss"].tolist()
            train_iou_history = history["train_iou"].tolist()
            val_loss_history = history["val_loss"].tolist()
            val_dice_history = history["val_dice"].tolist()
            val_iou_history = history["val_iou"].tolist()
            val_hd95_history = history["val_hd95"].tolist()
            best_iou = max(val_iou_history) if val_iou_history else 0.0
            start_epoch = len(train_loss_history)
            print(f"Loaded training history: {start_epoch} epochs, best IoU: {best_iou:.4f}")

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Starting training from epoch {start_epoch + 1} to {num_epochs}")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, num_epochs):
        epoch_idx = epoch + 1
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch_idx}/{num_epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.7f}")
        print(f"{'=' * 60}")

        # Training
        epoch_train_loss, epoch_train_iou = train_epoch(
            model, train_dataloader, optimizer, seg_loss, device, epoch_idx, num_epochs
        )

        train_loss_history.append(epoch_train_loss)
        train_iou_history.append(epoch_train_iou)

        print(f"\nTraining Results:")
        print(f"  Loss: {epoch_train_loss:.4f}")
        print(f"  IoU:  {epoch_train_iou:.4f}")

        # Validation
        val_metrics = validate(model, val_dataloader, seg_loss, device)

        val_loss_history.append(val_metrics['loss'])
        val_dice_history.append(val_metrics['dice'])
        val_iou_history.append(val_metrics['iou'])
        val_hd95_history.append(val_metrics['hd95'])

        print(f"\nValidation Results:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Dice: {val_metrics['dice']:.4f}")
        print(f"  IoU:  {val_metrics['iou']:.4f}")
        print(f"  HD95: {val_metrics['hd95']:.2f}" if not np.isnan(
            val_metrics['hd95']) else "  HD95: N/A")

        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            no_improve_epochs = 0

            rank = config_file["SAM"]["RANK"]
            model_path = os.path.join(
                exp_dir,
                f"best_model_rank{rank}_lru{args.lru_hidden_dim}_epoch{epoch_idx}.pth"
            )
            model.save_checkpoint(model_path)
            print(f"\nNew best model saved: {model_path}")
            print(f"Best IoU: {best_iou:.4f}")
        else:
            no_improve_epochs += 1
            print(f"\nNo improvement for {no_improve_epochs}/{patience} epochs")

        # Save periodic checkpoint
        if checkpoint_freq > 0 and (epoch_idx % checkpoint_freq == 0 or epoch_idx == num_epochs):
            rank = config_file["SAM"]["RANK"]
            ckpt_path = os.path.join(
                exp_dir, "checkpoints",
                f"checkpoint_rank{rank}_lru{args.lru_hidden_dim}_epoch{epoch_idx}.pth"
            )
            model.save_checkpoint(ckpt_path)

            # Save training history
            history_path = os.path.join(exp_dir, "training_history.npz")
            np.savez(history_path,
                     train_loss=np.array(train_loss_history),
                     train_iou=np.array(train_iou_history),
                     val_loss=np.array(val_loss_history),
                     val_dice=np.array(val_dice_history),
                     val_iou=np.array(val_iou_history),
                     val_hd95=np.array(val_hd95_history))

            print(f"Checkpoint saved: {ckpt_path}")

        # Early stopping
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch_idx}")
            break

    # Save final model
    rank = config_file["SAM"]["RANK"]
    final_model_path = os.path.join(
        exp_dir,
        f"final_model_rank{rank}_lru{args.lru_hidden_dim}_epoch{epoch_idx}.pth"
    )
    model.save_checkpoint(final_model_path)
    print(f"\nFinal model saved: {final_model_path}")

    # Plot training metrics
    plt.figure(figsize=(15, 12))

    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(train_loss_history, label='Training Loss', marker='o')
    plt.plot(val_loss_history, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.legend()

    # IoU plot
    plt.subplot(2, 2, 2)
    plt.plot(train_iou_history, label='Training IoU', color='orange', marker='o')
    plt.plot(val_iou_history, label='Validation IoU', color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training and Validation IoU')
    plt.grid(True)
    plt.legend()

    # Dice plot
    plt.subplot(2, 2, 3)
    plt.plot(val_dice_history, label='Validation Dice', marker='s')
    plt.plot(val_iou_history, label='Validation IoU', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Dice and IoU')
    plt.grid(True)
    plt.legend()

    # HD95 plot
    plt.subplot(2, 2, 4)
    plt.plot(val_hd95_history, label='Validation HD95', color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('HD95')
    plt.title('Validation HD95')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(exp_dir, "training_metrics.png")
    plt.savefig(plot_path)
    print(f"Training metrics plot saved: {plot_path}")

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Results saved to: {exp_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
