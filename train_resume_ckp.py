"""
首次训练
surgical_semantic_seg/benmarking_algorithms/Sam_LoRA/train.py

继续训练（加载 LoRA + mask_decoder 权重后继续，训练过程与 train.py 完全一致）：
1. check 'Current learning rate' of trainX.log and modify LR in config_resume.yaml
2. check the epoch number in the lora weights filename and set --lora-weights accordingly
3. check the experiment id and set --experiment-id to avoid overwriting previous results
4. 修改数据折：把下面的 mode="trainX"/"valX" 改成正在继续训练的那一折 (e.g. train2/val2)
5. 如果之前训练没有保存training history .npz文件，则从之前训练的log中获取best_iou (search "New best model saved with IoU")，
    grep "New best model saved with IoU" train5.log | tail -1
    设置 --best-iou 参数
6. change the trainX_Xepoch.log filename accordingly (trainX: experiment id, Xepoch: epoch number in the lora weights filename)
CUDA_VISIBLE_DEVICES=? nohup poetry run python train_resume_ckp.py \
    --config config_resume.yaml \
    --resume --best-iou ? \
    --lora-weights /mnt/hdd2/task2/sam_lora/exp_5/lora_rank2_22_epoch_in_100_epochs_final_5.safetensors \
    --experiment-id 6 \
    > /mnt/hdd2/task2/sam_lora/train6_resume_22epoch.log 2>&1 &

加载说明：
save_lora_parameters 会把 LoRA(A/B) 和 mask_decoder 的权重一起存进同一个 safetensors。
load_lora_parameters 会对应地把两者都原地加载回来 —— 因此只需调用一次
sam_lora.load_lora_parameters(path) 即可同时恢复 LoRA 和 decoder。

Note:
But this resumes model + decoder weights but not the Adam optimizer state (momentum buffers) since did not save optimizer.state_dict().
Influence: 
no optimizer state, LR (set it manually in config_resume.yaml) and dataloader shuffling restarts.

"""

import numpy as np
import torch
from medpy import metric
import monai
import yaml
import os
import re
import argparse
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('/home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA')
from src.utils import stacking_batch
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b
from src.lora import LoRA_sam
import matplotlib.pyplot as plt
from torch.optim import Adam, Optimizer
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
import warnings


# 配置参数解析器
def parse_args():
    parser = argparse.ArgumentParser(description='SAM-LoRA Training (resume)')
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--best_iou', type=float, default=0.0, help='Best IoU from previous training (if no history file found)')
    parser.add_argument('--lora_weights', type=str, default='', help='Path to LoRA(+decoder) weights for resuming')
    parser.add_argument('--experiment_id', type=int, default=1, help='Experiment ID for logging')
    return parser.parse_args()


# 计算指标函数
def calculate_metrics(pred, target):
    """
    Calculate Intersection over Union (IoU) between two binary masks.

    Arguments:
        pred: Predicted mask (tensor)
        target: Ground truth mask (tensor)

    Returns:
        dice, iou, hd95
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
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Arguments:
        pred: Predicted mask (tensor)
        target: Ground truth mask (tensor)
    
    Returns:
        iou: IoU score (tensor scalar)
        hd95
    """
    pred_binary = (pred > 0.5).float()
    label_binary = (target > 0.5).float()
    pred_binary = pred_binary.cpu().numpy().astype(bool)
    label_binary = label_binary.cpu().numpy().astype(bool)

    intersection = np.logical_and(pred_binary, label_binary)
    union = np.logical_or(pred_binary, label_binary)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    
    return iou


def main():
    args = parse_args()

    # 加载配置文件
    with open(args.config, "r") as ymlfile:
        config_file = yaml.safe_load(ymlfile)

    num_exp = args.experiment_id
    # 创建实验目录
    exp_dir = os.path.join("/mnt/hdd2/task2/sam_lora", f"exp_{num_exp}")
    os.makedirs(exp_dir, exist_ok=True)

    rank = config_file["SAM"]["RANK"]

    # ---------------------------------------------------------------------
    # 构建模型（与 train.py 一致）
    # ---------------------------------------------------------------------
    train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]
    sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
    sam_lora = LoRA_sam(sam, rank)
    model = sam_lora.sam
    processor = Samprocessor(model)

    # Create train dataloader —— 记得改成正在继续训练的那一折
    train_ds = DatasetSegmentation(config_file, processor, mode="train3")
    train_dataloader = DataLoader(train_ds,
                                  batch_size=config_file["TRAIN"]["BATCH_SIZE"],
                                  shuffle=True,
                                  collate_fn=collate_fn)

    # Create val dataloader —— 记得改成正在继续训练的那一折
    val_ds = DatasetSegmentation(config_file, processor, mode="val3")
    val_dataloader = DataLoader(val_ds,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=collate_fn)  # 验证集batch_size=1确保样本级评估

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 先把模型放到目标设备上，再加载权重（load 是原地拷贝，会保持设备）
    model.train()
    model.to(device)

    # ---------------------------------------------------------------------
    # 恢复训练状态
    # ---------------------------------------------------------------------
    start_epoch = 0
    best_iou = 0.0
    patience = config_file["TRAIN"].get("PATIENCE", 5)
    no_improve_epochs = 0

    train_loss_history = []
    train_iou_history = []
    val_loss_history = []
    val_dice_history = []
    val_iou_history = []
    val_hd95_history = []

    if args.resume:
        print(f"Resuming training from LoRA(+decoder) weights: {args.lora_weights}")
        if not os.path.exists(args.lora_weights):
            raise FileNotFoundError(f"LoRA weights not found: {args.lora_weights}")

        print(f"Get best_iou from args: {args.best_iou}")
        best_iou = args.best_iou  # 如果没有历史文件，则从 args 开始

        # 一次性加载 LoRA + mask_decoder（原地拷贝，保持在 device 上）
        sam_lora.load_lora_parameters(args.lora_weights)
        print(f"Loaded LoRA + decoder weights from {args.lora_weights} to device {device}")

        # 解析已训练的epoch数（文件名里 _{epoch}_epoch）
        match = re.search(r'_(\d+)_epoch', args.lora_weights)
        if match:
            start_epoch = int(match.group(1))
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Warning: Could not determine epoch from filename. Starting from epoch 0.")

        # 尝试加载之前的训练历史
        history_path = os.path.join(exp_dir, "training_history.npz")
        if os.path.exists(history_path):
            history = np.load(history_path)
            train_loss_history = history["train_loss"].tolist()
            train_iou_history = history["train_iou"].tolist()
            val_loss_history = history["val_loss"].tolist()
            val_dice_history = history["val_dice"].tolist()
            val_iou_history = history["val_iou"].tolist()
            val_hd95_history = history["val_hd95"].tolist()
            best_iou = max(val_iou_history) if val_iou_history else args.best_iou
            print(f"Loaded training history with {len(train_loss_history)} epochs, best_iou={best_iou:.4f}")
        
    # ---------------------------------------------------------------------
    # 初始化优化器（在加载权重之后创建，保证引用的是最终的参数对象）
    # 与 train.py 保持一致：同时优化 image_encoder(LoRA) 和 mask_decoder
    # ---------------------------------------------------------------------
    lr = config_file["TRAIN"]["LR"]
    optimizer = Adam(
        list(model.image_encoder.parameters()) + list(model.mask_decoder.parameters()),
        lr=lr,
        weight_decay=config_file["TRAIN"].get("WEIGHT_DECAY", 0.0)
    )

    # ---------------------------------------------------------------------
    # 训练循环（与 train.py 完全一致，只是 epoch 从 start_epoch 开始）
    # ---------------------------------------------------------------------
    epoch_idx = start_epoch  # 防止循环一次都不跑时 epoch_idx 未定义
    for epoch in range(start_epoch, num_epochs):
        epoch_idx = epoch + 1
        epoch_losses = []
        epoch_ious = []
        print(f'EPOCH: {epoch}')
        print(f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=5)}")

        model.train()

        for i, batch in enumerate(tqdm(train_dataloader)):

            outputs = model(batched_input=batch,
                            multimask_output=False)

            stk_gt, stk_out = utils.stacking_batch(batch, outputs)
            stk_out = stk_out.squeeze(1)
            stk_gt = stk_gt.unsqueeze(1)  # We need to get the [B, C, H, W] starting from [H, W]
            loss = seg_loss(stk_out, stk_gt.float().to(device))
            iou = calculate_iou(stk_out, stk_gt.float().to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_ious.append(iou)

        # 记录训练指标
        epoch_train_loss = mean(epoch_losses)
        epoch_train_iou = mean(epoch_ious)
        train_loss_history.append(epoch_train_loss)
        train_iou_history.append(epoch_train_iou)

        print(f'EPOCH: {epoch}, Training Mean loss: {epoch_train_loss}, Training Mean iou: {epoch_train_iou}')

        # Validation processing
        model.eval()
        val_losses = []
        val_dices = []
        val_ious = []
        val_hd95s = []

        with torch.no_grad():
            for val_batch in tqdm(val_dataloader, desc="Validating"):
                outputs = model(batched_input=val_batch, multimask_output=False)

                stk_gt, stk_out = utils.stacking_batch(val_batch, outputs)
                stk_out = stk_out.squeeze(1)
                stk_gt = stk_gt.unsqueeze(1)

                val_loss = seg_loss(stk_out, stk_gt.float().to(device))
                dice, iou, hd95 = calculate_metrics(stk_out, stk_gt.float().to(device))

                val_losses.append(val_loss.item())
                val_dices.append(dice)
                val_ious.append(iou)
                val_hd95s.append(hd95)

        # 处理NaN值（HD95计算可能产生NaN）
        valid_hd95s = [x for x in val_hd95s if not np.isnan(x)]
        val_mean_hd95 = np.mean(valid_hd95s) if valid_hd95s else np.nan
        val_mean_loss = np.mean(val_losses)
        val_mean_dice = np.mean(val_dices)
        val_mean_iou = np.mean(val_ious)

        # 记录验证指标
        val_loss_history.append(val_mean_loss)
        val_dice_history.append(val_mean_dice)
        val_iou_history.append(val_mean_iou)
        val_hd95_history.append(val_mean_hd95)

        # Print validation metrics
        print(f'Validation Mean - Loss: {val_mean_loss}, Dice: {val_mean_dice}, IoU: {val_mean_iou}, HD95: {val_mean_hd95}')
        print(f'Val sample counts - Total: {len(val_dices)}, Valid HD95: {len(valid_hd95s)}')

        # 保存最佳模型（命名与 train.py 一致）
        if val_mean_iou > best_iou:
            best_iou = val_mean_iou
            no_improve_epochs = 0
            sam_lora.save_lora_parameters(os.path.join(
                exp_dir, f"lora_rank{rank}_{epoch_idx}_epoch_in_{num_epochs}_epochs_best_{num_exp}.safetensors"
            ))
            print(f"New best model saved with IoU: {best_iou:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement in IoU for {no_improve_epochs}/{patience} epochs")

        # 每个 epoch 都保存训练历史，便于下次继续训练
        history_path = os.path.join(exp_dir, "training_history.npz")
        np.savez(history_path,
                 train_loss=np.array(train_loss_history),
                 train_iou=np.array(train_iou_history),
                 val_loss=np.array(val_loss_history),
                 val_dice=np.array(val_dice_history),
                 val_iou=np.array(val_iou_history),
                 val_hd95=np.array(val_hd95_history))

        # 早停检查
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Save the parameters of the model in safetensors format（命名与 train.py 一致）
    sam_lora.save_lora_parameters(os.path.join(
        exp_dir, f"lora_rank{rank}_{epoch_idx}_epoch_in_{num_epochs}_epochs_final_{num_exp}.safetensors"
    ))
    print(f"Final model saved after {epoch_idx} epochs")

    # 保存训练历史
    history_path = os.path.join(exp_dir, "training_history.npz")
    np.savez(history_path,
             train_loss=np.array(train_loss_history),
             train_iou=np.array(train_iou_history),
             val_loss=np.array(val_loss_history),
             val_dice=np.array(val_dice_history),
             val_iou=np.array(val_iou_history),
             val_hd95=np.array(val_hd95_history))

    # 可视化训练和验证指标
    plt.figure(figsize=(15, 12))

    # 子图1: 训练和验证损失
    plt.subplot(2, 2, 1)
    plt.plot(train_loss_history, label='Training Mean Loss', marker='o')
    plt.plot(val_loss_history, label='Validation Mean Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Mean Loss')
    plt.grid(True)
    plt.legend()

    # 子图2: 训练和验证IoU
    plt.subplot(2, 2, 2)
    plt.plot(train_iou_history, label='Training Mean IoU', color='orange', marker='o')
    plt.plot(val_iou_history, label='Validation Mean IoU', color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training and Validation Mean IoU')
    plt.grid(True)
    plt.legend()

    # 子图3: 验证Dice和IoU
    plt.subplot(2, 2, 3)
    plt.plot(val_dice_history, label='Validation Mean Dice', marker='s')
    plt.plot(val_iou_history, label='Validation Mean IoU', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Mean Dice and Mean IoU')
    plt.grid(True)
    plt.legend()

    # 子图4: 验证HD95
    plt.subplot(2, 2, 4)
    plt.plot(val_hd95_history, label='Validation Mean HD95', color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('HD95')
    plt.title('Validation Mean HD95')
    plt.grid(True)
    plt.legend()

    # 调整布局并保存
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, f"training_metrics_{num_exp}.png")
    plt.savefig(plot_path)
    print(f"Training metrics plot saved to {plot_path}")


if __name__ == "__main__":
    main()
