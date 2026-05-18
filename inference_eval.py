import csv
import os
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
import monai
import numpy as np
from medpy import metric
"""
The new evaluation version is '/home/lq/Projects_qin/surgical_semantic_seg/dataset_processed/utils_tools/inference_5fold_class_level_eval.py'

This file compute the evaluation metric (Dice cross entropy loss) for all trained LoRA SAM with different ranks. This gives the plot that is in ./plots/rank_comparison.jpg
which compares the performances on test the test set.

1. modify 
lora_mode = 
num_patient = 
num_fold =
safetensors_path = 

num_exp-training csv files pairs:
[
num_fold = 0 num_exp = 3 train1.csv
num_fold = 1 num_exp = 6 train2.csv
num_fold = 2 num_exp = 5 train3.csv
num_fold = 3 num_exp = 7 train4.csv
num_fold = 4 num_exp = 8 train5.csv
]

2. modify the path of log file '/exp_5/eval_0fold_19_final.log' and run:
CUDA_VISIBLE_DEVICES=0 nohup poetry run python inference_eval.py \
> /mnt/hdd2/task2/sam_lora/exp_9/eval_0fold_19_final.log 2>&1 &
"""

# ind = 0

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
    # global ind

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
            # hd95 = metric.binary.hd95(pred_binary, label_binary)
            hd95 = metric.binary.hd95(pred_binary.squeeze(), label_binary.squeeze())
        else:
            hd95 = np.nan
    except:
        hd95 = np.nan

    # # Save masks as images
    # pred_np = (pred > 0.5).float().cpu().numpy().squeeze()
    # target_np = (target > 0.5).float().cpu().numpy().squeeze()
    # # 1. Predicted mask
    # plt.figure(figsize=(6, 6))
    # plt.imshow(pred_np, cmap='gray')
    # plt.axis('off')
    # plt.title(f'Predicted Mask (Dice: {dice:.2f}, IoU: {iou:.2f})')
    # plt.savefig(os.path.join(fig_path, f'predicted_mask_{ind}.png'), bbox_inches='tight', pad_inches=0)
    # plt.close()
    
    # # 2. Ground truth mask
    # plt.figure(figsize=(6, 6))
    # plt.imshow(target_np, cmap='gray')
    # plt.axis('off')
    # plt.title('Ground Truth Mask')
    # plt.savefig(os.path.join(fig_path, f'ground_truth_mask_{ind}.png'), bbox_inches='tight', pad_inches=0)
    # plt.close()
    
    
    # ind += 1
    return dice, iou, hd95

lora_mode = "final" # "best"
num_patient = 19 # 19 24 71 76 78
num_fold = 0 # [0 1 2 3 4] for 5 folds; 'all' for training on all data

# safetensors_path = f"/home/lq/Projects_qin/surgical_semantic_seg/experiments/SAM_LoRA/experiment_2/{lora_mode}_model_rank2_7_epoch_in100epochs.safetensors"
# 0 fold
safetensors_path = f"/mnt/hdd2/task2/sam_lora/exp_3/lora_rank2_35_epoch_in_100_epochs_final_3.safetensors"
# 1 fold
# safetensors_path = f"/mnt/hdd2/task2/sam_lora/exp_6/lora_rank2_15_epoch_in_100_epochs_final_6.safetensors"
# 2 fold
# safetensors_path = f"/mnt/hdd2/task2/sam_lora/exp_5/lora_rank2_22_epoch_in_100_epochs_final_5.safetensors"
# 3 fold
# safetensors_path = f"/mnt/hdd2/task2/sam_lora/exp_7/lora_rank2_24_epoch_in_100_epochs_final_7.safetensors"
# 4 fold
# safetensors_path = f"/mnt/hdd2/task2/sam_lora/exp_8/lora_rank2_27_epoch_in_100_epochs_final_8.safetensors"

device = "cuda" if torch.cuda.is_available() else "cpu"
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
# Load the config file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

rank_list = [2] # [2, 4, 6, 8, 16, 32, 64, 128, 256, 512]

results = {
    "Baseline": {
        'Sample_Index': [],
        "loss": [],
        "dice": [],
        "iou": [],
        "iou_pred": [],
        "hd95": []
    }
}

for rank in rank_list:
    results[f"Rank {rank}"] = {
        'Sample_Index': [],
        "loss": [],
        "dice": [],
        "iou": [],
        "iou_pred": [],
        "hd95": []
    }

fig_path = f"/mnt/hdd2/task2/sam_lora/eval_{num_fold}fold_{num_patient}_{lora_mode}" # _without_bbox

csv_path = f"{fig_path}/results_inf_eval_rank{rank_list[0]}.csv"

os.makedirs(fig_path, exist_ok=True)

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model', 'Sample_Index', 'Loss', 'Dice', 'IoU', 'Predicted_IoU', 'HD95'])

# Load SAM model
with torch.no_grad():
    sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
    baseline = sam
    processor = Samprocessor(baseline)
    dataset = DatasetSegmentation(config_file, processor, mode="test")
    test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    baseline.eval()
    baseline.to(device)   
    for i, batch in enumerate(tqdm(test_dataloader)):
        
        outputs = baseline(batched_input=batch,
            multimask_output=False)
        
        gt_mask_tensor = batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0) # We need to get the [B, C, H, W] starting from [H, W]
        loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))
        iou_predictions = outputs[0]['iou_predictions']
        mask = outputs[0]["masks"]
        dice, iou_value, hd95 = calculate_metrics(mask.float(), gt_mask_tensor.float())
        
        results["Baseline"]['Sample_Index'].append(i)
        results["Baseline"]["loss"].append(loss.item())
        results["Baseline"]["dice"].append(dice)
        results["Baseline"]["iou"].append(iou_value)
        results["Baseline"]["iou_pred"].append(iou_predictions.item())
        results["Baseline"]["hd95"].append(hd95)

        # 将结果写入CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Baseline', i, loss.item(), dice, iou_value, 
                iou_predictions.item(), hd95
            ])

    print(f'Mean dice score: {mean(results["Baseline"]["dice"])}')
    baseline_loss = mean(results["Baseline"]["dice"])
    print(f'Mean IoU: {mean(results["Baseline"]["iou"])}')
    baseline_iou = mean(results["Baseline"]["iou"])
    print(f'Mean IoU Predictions: {mean(results["Baseline"]["iou_pred"])}')
    baseline_iou_predictions = mean(results["Baseline"]["iou_pred"])
    print(f'Mean HD95: {mean(results["Baseline"]["hd95"])}')
    baseline_hd95 = mean(results["Baseline"]["hd95"])

    for rank in rank_list:
        print(f"\nEvaluating Rank {rank} model...")
        sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
        # baseline = sam
        # Create SAM LoRA
        sam_lora = LoRA_sam(sam, rank)
        sam_lora.load_lora_parameters(safetensors_path)  
        model = sam_lora.sam
        
        # Process the dataset
        processor = Samprocessor(model)
        dataset = DatasetSegmentation(config_file, processor, mode="test")

        # Create a dataloader
        test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)


        # Set model to train and into the device
        model.eval()
        model.to(device)
    

        for i, batch in enumerate(tqdm(test_dataloader)):
            
            outputs = model(batched_input=batch,
                multimask_output=False)
            
            gt_mask_tensor = batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0) # We need to get the [B, C, H, W] starting from [H, W]
            loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))
            iou_predictions = outputs[0]['iou_predictions']
            mask = outputs[0]["masks"]
            dice, iou_value, hd95 = calculate_metrics(mask.float(), gt_mask_tensor.float())
            
            results[f"Rank {rank}"]['Sample_Index'].append(i)
            results[f"Rank {rank}"]["loss"].append(loss.item())
            results[f"Rank {rank}"]["dice"].append(dice)
            results[f"Rank {rank}"]["iou"].append(iou_value)
            results[f"Rank {rank}"]["iou_pred"].append(iou_predictions.item())
            results[f"Rank {rank}"]["hd95"].append(hd95)

            # 将结果写入CSV
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    f'Rank {rank}', i, loss.item(), dice, iou_value, 
                    iou_predictions.item(), hd95
                ])

        print(f'Mean dice score: {mean(results[f"Rank {rank}"]["dice"])}')
        print(f'Mean IoU: {mean(results[f"Rank {rank}"]["iou"])}')
        print(f'Mean IoU Predictions: {mean(results[f"Rank {rank}"]["iou_pred"])}')
        print(f'Mean HD95: {mean(results[f"Rank {rank}"]["hd95"])}')




# print("RANK LOSS :", rank_loss)

# width = 0.25  # the width of the bars
# multiplier = 0
# models_results= {"Baseline": baseline_loss,
#                  "Rank 2": rank_loss[0], 
#                 #  "Rank 4": rank_loss[1], 
#                 #  "Rank 6": rank_loss[2],
#                 #  "Rank 8": rank_loss[3],
#                 #  "Rank 16": rank_loss[4],
#                 #  "Rank 32": rank_loss[5],
#                 #  "Rank 64": rank_loss[6],
#                 #  "Rank 128": rank_loss[7],
#                 #  "Rank 256": rank_loss[8],
#                 #  "Rank 512": rank_loss[9]
#                  }
# eval_scores_name = ["Rank"]
# x = np.arange(len(eval_scores_name))
# fig, ax = plt.subplots(layout='constrained')

# for model_name, score in models_results.items():
#     offset = width * multiplier
#     rects = ax.bar(x + offset, score, width, label=model_name)
#     ax.bar_label(rects, padding=3)
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Dice Loss')
# ax.set_title('LoRA (Rank 2) trained on 12 epochs - Rank comparison on test set')
# ax.set_xticks(x + width, eval_scores_name)
# ax.legend(loc=3, ncols=2)
# ax.set_ylim(0, 0.2)

# plt.savefig(f"{fig_path}/rank_comparison.jpg")