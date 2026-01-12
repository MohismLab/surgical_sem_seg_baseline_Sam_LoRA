import os
import numpy as np
from medpy import metric
import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F
"""
This file is used to train a LoRA_sam model. I use that monai DiceLoss for the training. The batch size and number of epochs are taken from the configuration file.
The model is saved at the end as a safetensor.

每次训练前：
1. 修改 /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA/config.yaml
2. 修改下面代码的实验编号 num_exp = 1，如果是5折训练，则修改如下两行代码中的mode为对应csv的名称（["train1", "train2", "train3", "train4", "train5"], ["val1", "val2", "val3", "val4", "val5"]）：
train_ds = DatasetSegmentation(config_file, processor, mode="train")
val_ds = DatasetSegmentation(config_file, processor, mode="val")
3. 修改train{}.log：
CUDA_VISIBLE_DEVICES=? nohup poetry run python train.py > /home/lq/Projects_qin/surgical_semantic_seg/proposed_algorithm/SAM_LoRA/train1.log 2>&1 &
进程结束后请将log归类到相应的文件夹：mv /home/lq/Projects_qin/surgical_semantic_seg/proposed_algorithm/SAM_LoRA/train3.log /mnt/hdd2/task2/sam_lora/exp_3/
4. 预测：
CUDA_VISIBLE_DEVICES=? nohup poetry run python inference_eval.py
> /home/lq/Projects_qin/surgical_semantic_seg/proposed_algorithm/SAM_LoRA/inf_eval1.log 2>&1 &

可视化：
CUDA_VISIBLE_DEVICES=? nohup poetry run python inference_plots.py
> /home/lq/Projects_qin/surgical_semantic_seg/proposed_algorithm/SAM_LoRA/inf_plot1.log 2>&1 &

# dataset---------------------------------------------------------------------------------
# original: 
# /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA/dataset/train
# /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA/dataset/test
# 修改/home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA/config.yaml
# train/images
# train/masks (二值mask)
# 每个文件夹里面都是一个image和一个binary_mask一一对应
# 【但是这个mask是所有物体都在一张二值图】
# 【把每张图的所有类都单独抠出来了】
"""

# 实验编号
# num_exp = 3
##################################################################################
# 1月12号 2、3折训练 num_exp = 3 改为 4、5
num_exp = 5
##################################################################################

exp_dir = os.path.join("/mnt/hdd2/task2/sam_lora", f"exp_{num_exp}")
os.makedirs(exp_dir, exist_ok=True)

def calculate_metrics(pred, target):
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

# Load the config file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Take dataset path
train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]
# Load SAM model
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
#Create SAM LoRA
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])  
model = sam_lora.sam
# Process the dataset
processor = Samprocessor(model)

# Create train dataloader
# train_ds = DatasetSegmentation(config_file, processor, mode="train1")
##################################################################################
# 1月12号 2、3折训练 train2、3
train_ds = DatasetSegmentation(config_file, processor, mode="train3")
##################################################################################
train_dataloader = DataLoader(train_ds, batch_size=config_file["TRAIN"]["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)

# Create val dataloader
# val_ds = DatasetSegmentation(config_file, processor, mode="val1")
##################################################################################
# 1月12号 2、3折训练 val2、3
val_ds = DatasetSegmentation(config_file, processor, mode="val3")
##################################################################################
val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)  # 验证集batch_size=1确保样本级评估

# Initialize optimize and Loss
optimizer = Adam(model.image_encoder.parameters(), lr=1e-4, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

device = "cuda" if torch.cuda.is_available() else "cpu"
# Set model to train and into the device
model.train()
model.to(device)


# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, 
#     mode='max', 
#     factor=0.5, 
#     patience=2, 
#     verbose=True
# )

# 早停和模型保存相关变量
best_iou = 0.0
patience = 5
no_improve_epochs = 0

total_loss = []

# 初始化指标记录器
train_loss_history = []
train_iou_history = []
val_loss_history = []
val_dice_history = []
val_iou_history = []
val_hd95_history = []

for epoch in range(num_epochs):
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
      stk_gt = stk_gt.unsqueeze(1) # We need to get the [B, C, H, W] starting from [H, W]
      loss = seg_loss(stk_out, stk_gt.float().to(device))
      iou = calculate_iou(stk_out, stk_gt.float().to(device))

      optimizer.zero_grad()
      loss.backward()
      # optimize
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

    
    # 更新学习率
    # scheduler.step(mean_iou)
    
    # 保存最佳模型
    if val_mean_iou > best_iou:
        best_iou = val_mean_iou
        no_improve_epochs = 0
        rank = config_file["SAM"]["RANK"]
        sam_lora.save_lora_parameters(os.path.join(
            exp_dir, f"lora_rank{rank}_{epoch+1}_epoch_in_{num_epochs}_epochs_best_{num_exp}.safetensors"
        ))
        print(f"New best model saved with IoU: {best_iou:.4f}")
    else:
        no_improve_epochs += 1
        print(f"No improvement in IoU for {no_improve_epochs}/{patience} epochs")
    
    # 早停检查
    if no_improve_epochs >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break


# Save the parameters of the model in safetensors format
rank = config_file["SAM"]["RANK"]
sam_lora.save_lora_parameters(os.path.join(
            exp_dir, f"lora_rank{rank}_{epoch+1}_epoch_in_{num_epochs}_epochs_final_{num_exp}.safetensors"
))
print(f"Final model saved after {epoch+1} epochs")

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
output_path = os.path.join(exp_dir, f"training_metrics_{num_exp}.png")
plt.savefig(output_path)
print(f"Training metrics visualization saved to {output_path}")