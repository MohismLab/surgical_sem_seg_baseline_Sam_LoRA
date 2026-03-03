"""
Inference Script for SAM-LoRA-LRU

This script performs inference using a trained SAM-LoRA-LRU model
and evaluates it on the test set.

Usage:
    CUDA_VISIBLE_DEVICES=0 python inference_SAM_LRU.py \
        --checkpoint path/to/model.pth \
        --config config.yaml \
        --output-dir ./results
"""

import numpy as np
import torch
import monai
import yaml
import os
import argparse
import csv
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from medpy import metric
import sys

sys.path.append('/home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA')

from src.dataloader_sequential import DatasetSegmentationSequential, collate_fn_sequential
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b
from SAM_LRU import LRU_SAM


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SAM-LoRA-LRU Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                        help='Directory to save results')
    parser.add_argument('--lru-hidden-dim', type=int, default=512,
                        help='Hidden dimension for LRU (must match training)')
    parser.add_argument('--test-mode', type=str, default='test',
                        help='Test mode (test/val2/etc)')
    parser.add_argument('--disable-lru', action='store_true',
                        help='Disable LRU (for baseline inference)')
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


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration
    with open(args.config, "r") as ymlfile:
        config_file = yaml.safe_load(ymlfile)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize base SAM model
    print("Loading SAM model...")
    sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])

    # Initialize LRU_SAM model
    print("Initializing LRU_SAM model...")
    model = LRU_SAM(
        sam_model=sam,
        lora_rank=config_file["SAM"]["RANK"],
        lru_hidden_dim=args.lru_hidden_dim,
        use_lru=not args.disable_lru
    )

    # Load trained weights
    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_checkpoint(args.checkpoint)

    # Move model to device
    model.sam.to(device)
    if model.lru is not None:
        model.lru.to(device)

    # Set to evaluation mode
    model.sam.eval()
    if model.lru is not None:
        model.lru.eval()

    # Create processor and dataset
    processor = Samprocessor(model.sam)
    print(f"Loading test dataset (mode: {args.test_mode})...")
    test_ds = DatasetSegmentationSequential(config_file, processor, mode=args.test_mode)
    test_dataloader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_sequential
    )

    print(f"Test dataset: {len(test_ds)} samples")

    # Get video sequences info
    video_sequences = test_ds.get_video_sequences()
    print(f"Number of videos: {len(video_sequences)}")

    # Loss function
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # Storage for results
    results = {
        'sample_index': [],
        'video_id': [],
        'frame_id': [],
        'loss': [],
        'dice': [],
        'iou': [],
        'iou_pred': [],
        'hd95': []
    }

    # CSV output path
    csv_path = os.path.join(args.output_dir, 'inference_results.csv')

    # Write CSV header
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sample_Index', 'Video_ID', 'Frame_ID', 'Loss', 'Dice', 'IoU', 'IoU_Pred', 'HD95'])

    # Inference loop
    print("\nRunning inference...")
    current_video_id = None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader)):
            # Check if starting new video
            batch_video_id = batch[0].get("video_id", None)
            batch_frame_id = batch[0].get("frame_id", None)

            if batch_video_id is not None and batch_video_id != current_video_id:
                # Reset hidden state for new video
                model.reset_hidden_state()
                current_video_id = batch_video_id
                print(f"\nProcessing video {current_video_id}")

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
            gt_mask_tensor = device_batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0)
            loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))
            iou_predictions = outputs[0]['iou_predictions']
            mask = outputs[0]["masks"]

            dice, iou_value, hd95 = calculate_metrics(mask.float(), gt_mask_tensor.float())

            # Store results
            results['sample_index'].append(i)
            results['video_id'].append(batch_video_id if batch_video_id is not None else -1)
            results['frame_id'].append(batch_frame_id if batch_frame_id is not None else -1)
            results['loss'].append(loss.item())
            results['dice'].append(dice)
            results['iou'].append(iou_value)
            results['iou_pred'].append(iou_predictions.item())
            results['hd95'].append(hd95)

            # Write to CSV
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    i,
                    batch_video_id if batch_video_id is not None else -1,
                    batch_frame_id if batch_frame_id is not None else -1,
                    loss.item(),
                    dice,
                    iou_value,
                    iou_predictions.item(),
                    hd95
                ])

    # Calculate mean metrics
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"Mean Loss:       {mean(results['loss']):.4f}")
    print(f"Mean Dice:       {mean(results['dice']):.4f}")
    print(f"Mean IoU:        {mean(results['iou']):.4f}")
    print(f"Mean IoU Pred:   {mean(results['iou_pred']):.4f}")

    # HD95 (excluding NaN values)
    valid_hd95 = [x for x in results['hd95'] if not np.isnan(x)]
    if valid_hd95:
        print(f"Mean HD95:       {mean(valid_hd95):.2f} ({len(valid_hd95)}/{len(results['hd95'])} valid)")
    else:
        print("Mean HD95:       N/A (no valid measurements)")

    print("=" * 60)

    # Calculate per-video statistics
    print("\nPer-Video Statistics:")
    print("-" * 60)

    for video_id in sorted(set(results['video_id'])):
        if video_id < 0:  # Skip invalid video IDs
            continue

        # Filter results for this video
        video_indices = [i for i, vid in enumerate(results['video_id']) if vid == video_id]
        video_dice = [results['dice'][i] for i in video_indices]
        video_iou = [results['iou'][i] for i in video_indices]

        print(f"Video {video_id}: {len(video_indices)} frames, "
              f"Dice={mean(video_dice):.4f}, IoU={mean(video_iou):.4f}")

    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SAM-LoRA-LRU Inference Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test mode: {args.test_mode}\n")
        f.write(f"Total samples: {len(results['sample_index'])}\n")
        f.write(f"Number of videos: {len([x for x in set(results['video_id']) if x >= 0])}\n")
        f.write("\nMean Metrics:\n")
        f.write(f"  Loss:     {mean(results['loss']):.4f}\n")
        f.write(f"  Dice:     {mean(results['dice']):.4f}\n")
        f.write(f"  IoU:      {mean(results['iou']):.4f}\n")
        f.write(f"  IoU Pred: {mean(results['iou_pred']):.4f}\n")
        if valid_hd95:
            f.write(f"  HD95:     {mean(valid_hd95):.2f}\n")

    print(f"\nResults saved to:")
    print(f"  CSV:     {csv_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
