import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict

import monai
import numpy as np
import pandas as pd
import torch
import yaml
from medpy import metric
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.utils as utils  # noqa: F401
from src.dataloader import DatasetSegmentation, collate_fn
from src.lora import LoRA_sam
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b

"""
The new evaluation version is '/home/lq/Projects_qin/surgical_semantic_seg/dataset_processed/utils_tools/inference_5fold_class_level_eval.py'
modify inference_eval.py, modify the shape of input of calculate_metrics().
CUDA_VISIBLE_DEVICES=0 nohup poetry run python inference_eval_5fold_test.py \
> /mnt/hdd2/task2/sam_lora/eval_new.log 2>&1 &
"""

FOLD_FINAL_CKPT = {
    0: "/mnt/hdd2/task2/sam_lora/exp_3/lora_rank2_35_epoch_in_100_epochs_final_3.safetensors",
    1: "/mnt/hdd2/task2/sam_lora/exp_6/lora_rank2_15_epoch_in_100_epochs_final_6.safetensors",
    2: "/mnt/hdd2/task2/sam_lora/exp_5/lora_rank2_22_epoch_in_100_epochs_final_5.safetensors",
    3: "/mnt/hdd2/task2/sam_lora/exp_7/lora_rank2_24_epoch_in_100_epochs_final_7.safetensors",
    4: "/mnt/hdd2/task2/sam_lora/exp_8/lora_rank2_27_epoch_in_100_epochs_final_8.safetensors",
}

TEST_JSON = "/mnt/hdd2/task2/sam_lora/output_bbox_test.json"
OUT_BASE = "/mnt/hdd2/task2/sam_lora"
RANK = 2

INSTRUMENT_CLASSES = set(range(1, 26))  # 1..25
ORGAN_CLASSES = {26, 27, 28}


def setup_logger(out_dir):
    log_path = os.path.join(out_dir, "eval.log")
    logger = logging.getLogger(f"sam_lora_eval_{out_dir}")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def calculate_metrics(pred, target):
    pred_binary = (pred > 0.5).float().cpu().numpy().astype(bool)
    label_binary = (target > 0.5).float().cpu().numpy().astype(bool)
    inter = np.logical_and(pred_binary, label_binary)
    union = np.logical_or(pred_binary, label_binary)
    denom = np.sum(pred_binary) + np.sum(label_binary) + 1e-8
    dice = float(2.0 * np.sum(inter) / denom)
    iou = float(np.sum(inter) / np.sum(union)) if np.sum(union) > 0 else 0.0
    try:
        if np.sum(pred_binary) > 0 and np.sum(label_binary) > 0:
            # hd95 = float(metric.binary.hd95(pred_binary, label_binary))
            hd95 = metric.binary.hd95(pred_binary.squeeze(), label_binary.squeeze())
        else:
            hd95 = float("nan")
    except Exception:
        hd95 = float("nan")
    return dice, iou, hd95


def build_prompt_index(json_path):
    """Iterate JSON in same order as DatasetSegmentation produces samples.
    Returns list of (image_name, class_id, mask_path) per prompt index."""
    with open(json_path) as f:
        data = json.load(f)
    test_data = data.get("test", {})
    rows = []
    for img_name in sorted(test_data.keys()):
        for info in test_data[img_name]:
            mask_path = info["mask_path"]
            m = re.search(r"_class(\d+)\.png", mask_path)
            cls = int(m.group(1)) if m else -1
            rows.append((img_name, cls, mask_path))
    return rows


def extract_patient_id(image_name):
    m = re.match(r"(\d+)_", image_name)
    return m.group(1) if m else "unknown"


def aggregate(out_dir, csv_path, prompt_meta, logger, fold):
    """Read per-prompt CSV + prompt_meta, produce per-patient/per-class/organ_instr/overall CSVs."""
    df = pd.read_csv(csv_path)
    df = df[df["Model"] == f"Rank {RANK}"].reset_index(drop=True)
    if len(df) != len(prompt_meta):
        logger.warning(
            f"CSV rows {len(df)} != JSON prompts {len(prompt_meta)}; "
            f"will trim to min length"
        )
    n = min(len(df), len(prompt_meta))
    df = df.iloc[:n].copy()
    meta_arr = prompt_meta[:n]
    df["image_name"] = [m[0] for m in meta_arr]
    df["class_id"] = [m[1] for m in meta_arr]
    df["patient_id"] = df["image_name"].apply(extract_patient_id)
    # NaN/inf guard
    df["HD95"] = df["HD95"].replace([np.inf, -np.inf], np.nan)

    # save the enriched per-prompt CSV (overwrite in place)
    df.to_csv(csv_path, index=False)

    # ---- per-patient ----
    pp = (
        df.groupby("patient_id")
        .agg(
            mean_iou=("IoU", "mean"),
            mean_dice=("Dice", "mean"),
            mean_hd95=("HD95", "mean"),
            n_prompts=("IoU", "size"),
        )
        .reset_index()
    )
    overall_row = pd.DataFrame(
        [{
            "patient_id": "Overall",
            "mean_iou": df["IoU"].mean(),
            "mean_dice": df["Dice"].mean(),
            "mean_hd95": df["HD95"].mean(),
            "n_prompts": len(df),
        }]
    )
    pp = pd.concat([pp, overall_row], ignore_index=True)
    pp.to_csv(os.path.join(out_dir, "result_per_patient.csv"), index=False)

    # ---- per-patient per-class ----
    pcc = (
        df.groupby(["patient_id", "class_id"])
        .agg(
            mean_iou=("IoU", "mean"),
            mean_dice=("Dice", "mean"),
            mean_hd95=("HD95", "mean"),
            n_prompts=("IoU", "size"),
        )
        .reset_index()
    )
    pcc.to_csv(
        os.path.join(out_dir, "result_per_patient_per_class.csv"), index=False
    )

    # ---- organ / instrument ----
    def grp(c):
        if c in INSTRUMENT_CLASSES:
            return "Instrument"
        if c in ORGAN_CLASSES:
            return "Organ"
        return "Other"

    df["Group"] = df["class_id"].apply(grp)
    oi = (
        df.groupby("Group")
        .agg(
            **{
                "Mean Dice": ("Dice", "mean"),
                "Mean IoU": ("IoU", "mean"),
                "Mean HD95": ("HD95", "mean"),
                "n_prompts": ("IoU", "size"),
            }
        )
        .reset_index()
    )
    order = ["Instrument", "Organ", "Other"]
    oi["_o"] = oi["Group"].apply(lambda g: order.index(g) if g in order else len(order))
    oi.sort_values("_o", inplace=True)
    oi.drop(columns=["_o"], inplace=True)
    oi.to_csv(os.path.join(out_dir, "result_organ_instrument.csv"), index=False)

    # ---- overall ----
    overall = pd.DataFrame(
        [{
            "fold": fold,
            "Mean IoU": df["IoU"].mean(),
            "Mean Dice": df["Dice"].mean(),
            "Mean HD95": df["HD95"].mean(),
            "n_prompts": len(df),
            "n_patients": df["patient_id"].nunique(),
            "n_classes": df["class_id"].nunique(),
        }]
    )
    overall.to_csv(os.path.join(out_dir, "result_overall.csv"), index=False)

    # ---- log summary ----
    logger.info("=" * 80)
    logger.info(f"Fold {fold} aggregation summary")
    logger.info("=" * 80)
    logger.info(
        f"Overall  mIoU={df['IoU'].mean():.5f}  "
        f"mDice={df['Dice'].mean():.5f}  "
        f"mHD95={df['HD95'].mean():.5f}  "
        f"n_prompts={len(df)}"
    )
    logger.info("-" * 80)
    logger.info("Per-patient:")
    for _, r in pp.iterrows():
        logger.info(
            f"  {str(r['patient_id']):>8}  "
            f"mIoU={r['mean_iou']:.5f}  "
            f"mDice={r['mean_dice']:.5f}  "
            f"mHD95={r['mean_hd95']:.5f}  "
            f"n={r['n_prompts']}"
        )
    logger.info("-" * 80)
    logger.info("Group:")
    for _, r in oi.iterrows():
        logger.info(
            f"  {r['Group']:>10}  "
            f"mIoU={r['Mean IoU']:.5f}  "
            f"mDice={r['Mean Dice']:.5f}  "
            f"mHD95={r['Mean HD95']:.5f}  "
            f"n={r['n_prompts']}"
        )
    logger.info("=" * 80)


def run_fold(fold, ckpt_path, config_file, prompt_meta, device, seg_loss, force=False):
    out_dir = os.path.join(OUT_BASE, f"eval_{fold}fold_test_final")
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger(out_dir)
    csv_path = os.path.join(out_dir, f"results_inf_eval_rank{RANK}.csv")

    logger.info("=" * 80)
    logger.info(f"FOLD {fold}")
    logger.info(f"  ckpt: {ckpt_path}")
    logger.info(f"  test json: {TEST_JSON}")
    logger.info(f"  out dir: {out_dir}")
    logger.info(f"  expected prompts: {len(prompt_meta)}")
    logger.info("=" * 80)

    if (not force) and os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        n_rank = (df_existing["Model"] == f"Rank {RANK}").sum() if "Model" in df_existing else 0
        if n_rank == len(prompt_meta):
            logger.info(f"CSV already complete ({n_rank} rank{RANK} rows). Skipping inference, re-running aggregation only.")
            aggregate(out_dir, csv_path, prompt_meta, logger, fold)
            return

    if not os.path.exists(ckpt_path):
        logger.error(f"ckpt not found: {ckpt_path}")
        return

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model", "Sample_Index", "Loss", "Dice", "IoU", "Predicted_IoU", "HD95"])

    sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
    sam_lora = LoRA_sam(sam, RANK)
    sam_lora.load_lora_parameters(ckpt_path)
    model = sam_lora.sam
    model.eval()
    model.to(device)

    processor = Samprocessor(model)
    dataset = DatasetSegmentation(config_file, processor, mode="test")
    test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    if len(dataset) != len(prompt_meta):
        logger.warning(
            f"Dataset size {len(dataset)} != JSON prompts {len(prompt_meta)}; "
            f"alignment may be off."
        )

    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc=f"fold {fold}")):
            outputs = model(batched_input=batch, multimask_output=False)
            gt_mask_tensor = batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0)
            loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))
            iou_pred = outputs[0]["iou_predictions"]
            mask = outputs[0]["masks"]
            dice, iou_value, hd95 = calculate_metrics(mask.float(), gt_mask_tensor.float())
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([f"Rank {RANK}", i, loss.item(), dice, iou_value, iou_pred.item(), hd95])

    elapsed = time.time() - t0
    logger.info(f"Inference done in {elapsed/60:.1f} min")

    aggregate(out_dir, csv_path, prompt_meta, logger, fold)

    del model, sam_lora, sam
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=str, default="0,1,2,3,4",
                        help="Comma-separated fold ids to run")
    parser.add_argument("--force", action="store_true",
                        help="Re-run inference even if CSV exists")
    args = parser.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    with open("./config.yaml", "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    prompt_meta = build_prompt_index(TEST_JSON)
    print(f"[init] expected {len(prompt_meta)} prompts from {TEST_JSON}")

    for fold in folds:
        if fold not in FOLD_FINAL_CKPT:
            print(f"[skip] fold {fold} has no registered ckpt")
            continue
        run_fold(fold, FOLD_FINAL_CKPT[fold], config_file, prompt_meta, device, seg_loss, force=args.force)


if __name__ == "__main__":
    main()
