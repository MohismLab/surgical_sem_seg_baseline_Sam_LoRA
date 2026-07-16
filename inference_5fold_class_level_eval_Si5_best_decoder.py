"""
Author: Zhouxuan Xia
5-Fold SAM-LoRA inference on the Si5 external test set — FINAL weights.
Forked from inference_5fold_class_level_eval_Si5.py (best → final).

Run:
    CUDA_VISIBLE_DEVICES=1 nohup \
        /home/lq/Projects_qin/run_SSAM/bin/python \
        /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA/inference_5fold_class_level_eval_Si5_final.py \
        > logs/inference_5fold_Si5_final.log 2>&1 &
"""

import json, os, sys, re, glob, time, csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from medpy import metric
from tqdm import tqdm
import logging

sys.path.insert(0, "/home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA/src")
sys.path.insert(0, "/home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA")
from lora import LoRA_sam
from processor import Samprocessor
from dataloader import collate_fn
from segment_anything import build_sam_vit_b
import utils as sam_lora_utils


# ============================================================
# Config
# ============================================================
SAM_CKPT = "/mnt/hdd2/task2/sam/sam_vit_b_01ec64.pth"
SI5_BASE = "/mnt/hdd2/task2/Si5/processed/sam_lora"
OUT_BASE = "/mnt/hdd2/task2/sam_lora/class_eval_5fold_Si5_best_decoder"   # ← best
RANK = 2
DEVICE = "cuda:0"
ALL_CLASSES = list(range(1, 29))
ORGAN = {26, 27, 28}
INSTR = set(range(1, 26))
TEST_PATIENTS = ["1", "2", "3", "4", "5"]

FOLD_CKPT = {
    0: "/mnt/hdd2/task2/sam_lora/exp_9/lora_rank2_25_epoch_in_100_epochs_best_9.safetensors",
    1: "/mnt/hdd2/task2/sam_lora/exp_10/lora_rank2_19_epoch_in_100_epochs_best_10.safetensors",
    2: "/mnt/hdd2/task2/sam_lora/exp_11/lora_rank2_43_epoch_in_100_epochs_best_11.safetensors",
    3: "/mnt/hdd2/task2/sam_lora/exp_12/lora_rank2_26_epoch_in_100_epochs_best_12.safetensors",
    4: "/mnt/hdd2/task2/sam_lora/exp_13/lora_rank2_34_epoch_in_100_epochs_best_13.safetensors",
}

os.makedirs(OUT_BASE, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUT_BASE, "eval.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ============================================================
# Si5 dataset (per-patient)
# ============================================================
class Si5Dataset(Dataset):
    """Per-patient Si5 test dataset (unchanged from best version)."""

    def __init__(self, processor, patient_id):
        super().__init__()
        self.processor = processor

        json_path = os.path.join(SI5_BASE, f"output_bbox_test_Si5_{patient_id}.json")
        img_dir = os.path.join(SI5_BASE, f"test_{patient_id}", "images")

        with open(json_path) as f:
            data = json.load(f)
        test_data = data.get("test", {})

        self.img_files = []
        self.mask_files = []
        self.prompt_index = []

        for img_name in sorted(test_data.keys()):
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                print(f"Warning: image not found - {img_path}")
                continue
            for info in test_data[img_name]:
                m = re.search(r"_class(\d+)\.png", info["mask_path"])
                cls_id = int(m.group(1)) if m else -1
                self.img_files.append(img_path)
                self.mask_files.append(info["mask_path"])
                self.prompt_index.append((img_name, cls_id))

        self.patient_id = patient_id

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = Image.open(img_path)
        mask = Image.open(mask_path).convert("1")
        ground_truth_mask = np.array(mask)
        original_size = tuple(image.size)[::-1]
        box = sam_lora_utils.get_bounding_box(ground_truth_mask)
        inputs = self.processor(image, original_size, box)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)
        return inputs


# ============================================================
# Per-fold inference
# ============================================================
def run_fold_inference(fold):
    out_dir = os.path.join(OUT_BASE, f"fold{fold}")
    pred_dir = os.path.join(out_dir, "pred_masks")
    os.makedirs(pred_dir, exist_ok=True)

    ckpt = FOLD_CKPT[fold]
    log.info(f"Fold {fold}: loading {ckpt}")

    sam = build_sam_vit_b(checkpoint=SAM_CKPT)
    sam_lora = LoRA_sam(sam, RANK)
    sam_lora.load_lora_parameters(ckpt)
    model = sam_lora.sam
    model.eval()
    model.to(DEVICE)

    processor = Samprocessor(model)

    existing_masks = set(os.listdir(pred_dir)) if os.path.exists(pred_dir) else set()
    if existing_masks:
        log.info(f"  Found {len(existing_masks)} existing pred masks, will skip already-saved")

    inst_results = []
    t_start = time.time()

    for pid in TEST_PATIENTS:
        dataset = Si5Dataset(processor, pid)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        total = len(dataloader)
        log.info(f"  Fold {fold} patient {pid}: {total} prompts")

        for i, batch in enumerate(tqdm(dataloader, desc=f"Fold{fold} P{pid}", total=total)):
            img_name, cls_id = dataset.prompt_index[i]
            base = img_name.replace(".png", "")
            pred_name = f"{base}_class{cls_id}.png"

            gt_mask = batch[0]["ground_truth_mask"].numpy().astype(bool)

            if pred_name in existing_masks:
                pred_bin = np.array(Image.open(os.path.join(pred_dir, pred_name))) > 0
            else:
                with torch.no_grad():
                    outputs = model(batched_input=batch, multimask_output=False)
                pred_mask = outputs[0]["masks"]
                pred_arr = (pred_mask > 0.5).cpu().numpy().squeeze().astype(np.uint8) * 255
                Image.fromarray(pred_arr).save(os.path.join(pred_dir, pred_name))
                pred_bin = pred_arr > 0

            inter = np.logical_and(pred_bin, gt_mask).sum()
            union = np.logical_or(pred_bin, gt_mask).sum()
            iou = inter / union if union > 0 else 0.0
            dice = 2 * inter / (pred_bin.sum() + gt_mask.sum() + 1e-8)
            try:
                hd = (
                    float(metric.binary.hd95(pred_bin, gt_mask))
                    if pred_bin.any() and gt_mask.any()
                    else np.nan
                )
            except Exception:
                hd = np.nan

            inst_results.append({
                "patient": pid, "img": img_name, "class": cls_id,
                "iou": iou, "dice": dice, "hd95": hd,
            })

    t_elapsed = (time.time() - t_start) / 60
    log.info(f"  Fold {fold} inference: {t_elapsed:.1f} min, {len(inst_results)} instances")
    return inst_results, pred_dir


# ============================================================
# Per-fold class-level eval
# ============================================================
def run_class_eval(fold, pred_dir):
    log.info(f"Fold {fold}: computing class-level metrics...")
    t_start = time.time()
    cls_results = []

    for pid in TEST_PATIENTS:
        json_path = os.path.join(SI5_BASE, f"output_bbox_test_Si5_{pid}.json")
        with open(json_path) as f:
            data = json.load(f)
        test_data = data.get("test", {})
        gt_mask_dir = os.path.join(SI5_BASE, f"test_{pid}", "masks")

        for img_name in tqdm(sorted(test_data.keys()),
                             desc=f"Fold{fold} ClassEval P{pid}", leave=False):
            base = img_name.replace(".png", "")
            pattern = f"{base}_class*.png"

            gt_map = np.zeros((1024, 1024), dtype=np.uint8)
            for fp in sorted(glob.glob(os.path.join(gt_mask_dir, pattern))):
                cls = int(os.path.basename(fp).replace(".png", "").rsplit("_class", 1)[1])
                gt_map[np.array(Image.open(fp)) > 0] = cls

            pred_map = np.zeros((1024, 1024), dtype=np.uint8)
            for fp in sorted(glob.glob(os.path.join(pred_dir, pattern))):
                cls = int(os.path.basename(fp).replace(".png", "").rsplit("_class", 1)[1])
                pred_map[np.array(Image.open(fp)) > 0] = cls

            for cls in ALL_CLASSES:
                gt_bin = (gt_map == cls)
                pred_bin = (pred_map == cls)
                if not gt_bin.any():
                    continue
                inter = (pred_bin & gt_bin).sum()
                union = (pred_bin | gt_bin).sum()
                iou = inter / union if union > 0 else 0.0
                dice = 2 * inter / (pred_bin.sum() + gt_bin.sum() + 1e-8)
                try:
                    hd = (
                        float(metric.binary.hd95(pred_bin, gt_bin))
                        if pred_bin.any() and gt_bin.any()
                        else np.nan
                    )
                except Exception:
                    hd = np.nan
                cls_results.append({
                    "patient": pid, "img": img_name, "class": cls,
                    "iou": iou, "dice": dice, "hd95": hd,
                })

    t_elapsed = (time.time() - t_start) / 60
    log.info(f"  Fold {fold} class eval: {t_elapsed:.1f} min, {len(cls_results)} per-class rows")
    return cls_results


def summarize(results_list):
    all_iou = [r["iou"] for r in results_list]
    all_dice = [r["dice"] for r in results_list]
    all_hd95 = [r["hd95"] for r in results_list if not np.isnan(r["hd95"])]
    org_hd95 = [r["hd95"] for r in results_list if r["class"] in ORGAN and not np.isnan(r["hd95"])]
    ins_hd95 = [r["hd95"] for r in results_list if r["class"] in INSTR and not np.isnan(r["hd95"])]
    org_iou = [r["iou"] for r in results_list if r["class"] in ORGAN]
    ins_iou = [r["iou"] for r in results_list if r["class"] in INSTR]
    org_dice = [r["dice"] for r in results_list if r["class"] in ORGAN]
    ins_dice = [r["dice"] for r in results_list if r["class"] in INSTR]
    return {
        "mIoU": float(np.mean(all_iou)) if all_iou else float("nan"),
        "mDice": float(np.mean(all_dice)) if all_dice else float("nan"),
        "mHD95": float(np.mean(all_hd95)) if all_hd95 else float("nan"),
        "Organ mIoU": float(np.mean(org_iou)) if org_iou else float("nan"),
        "Organ mDice": float(np.mean(org_dice)) if org_dice else float("nan"),
        "Organ HD95": float(np.mean(org_hd95)) if org_hd95 else float("nan"),
        "Organ n": len(org_iou),
        "Instr mIoU": float(np.mean(ins_iou)) if ins_iou else float("nan"),
        "Instr mDice": float(np.mean(ins_dice)) if ins_dice else float("nan"),
        "Instr HD95": float(np.mean(ins_hd95)) if ins_hd95 else float("nan"),
        "Instr n": len(ins_iou),
        "n": len(results_list),
    }


# ============================================================
# Main
# ============================================================
def main():
    log.info(f"5-Fold Class-Level Evaluation on Si5 — BEST weights. Patients: {TEST_PATIENTS}")
    log.info(f"Folds: {list(FOLD_CKPT.keys())}")

    all_fold_cls = {}

    for fold in range(5):
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold}")
        log.info(f"{'='*60}")

        inst_results, pred_dir = run_fold_inference(fold)
        cls_results = run_class_eval(fold, pred_dir)
        all_fold_cls[fold] = summarize(cls_results)

        csv_path = os.path.join(OUT_BASE, f"fold{fold}", "class_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient", "img", "class", "iou", "dice", "hd95"])
            for r in cls_results:
                writer.writerow([r["patient"], r["img"], r["class"], r["iou"], r["dice"], r["hd95"]])

        log.info(
            f"  Fold {fold}: class mIoU={all_fold_cls[fold]['mIoU']:.4f}, "
            f"class HD95={all_fold_cls[fold]['mHD95']:.2f}, "
            f"class n={all_fold_cls[fold]['n']}"
        )

    # 5-fold summary
    log.info(f"\n{'='*70}")
    log.info("5-FOLD SUMMARY: SAM_LoRA Class-Level Evaluation on Si5 (FINAL)")
    log.info(f"{'='*70}")
    log.info(
        f"{'Fold':<6} {'n':>6} {'mIoU':>8} {'mDice':>8} {'mHD95':>8} "
        f"{'O.IoU':>8} {'O.Dice':>8} {'O.HD95':>8} {'O.n':>6} "
        f"{'I.IoU':>8} {'I.Dice':>8} {'I.HD95':>8} {'I.n':>6}"
    )

    col_names = [
        "n_rows", "Mean IOU", "Mean Dice", "Mean HD95",
        "(Organ) Mean IOU", "(Organ) Mean Dice", "(Organ) Mean HD95", "(Organ) n",
        "(Instr) Mean IOU", "(Instr) Mean Dice", "(Instr) Mean HD95", "(Instr) n",
    ]
    int_cols = {"n_rows", "(Organ) n", "(Instr) n"}
    rows_for_csv = []

    for fold in range(5):
        c = all_fold_cls[fold]
        log.info(
            f"{fold:<6} {c['n']:>6} {c['mIoU']:>8.4f} {c['mDice']:>8.4f} {c['mHD95']:>8.2f} "
            f"{c['Organ mIoU']:>8.4f} {c['Organ mDice']:>8.4f} {c['Organ HD95']:>8.2f} "
            f"{c['Organ n']:>6} "
            f"{c['Instr mIoU']:>8.4f} {c['Instr mDice']:>8.4f} {c['Instr HD95']:>8.2f} "
            f"{c['Instr n']:>6}"
        )
        rows_for_csv.append([
            fold, c["n"], c["mIoU"], c["mDice"], c["mHD95"],
            c["Organ mIoU"], c["Organ mDice"], c["Organ HD95"], c["Organ n"],
            c["Instr mIoU"], c["Instr mDice"], c["Instr HD95"], c["Instr n"],
        ])

    mean_row = ["mean"]
    for i, col_name in enumerate(col_names):
        vals = [r[i + 1] for r in rows_for_csv]
        if col_name in int_cols:
            mean_row.append(int(np.mean(vals)))
        else:
            mean_row.append(float(np.mean(vals)))
    log.info(
        f"{mean_row[0]:<6} {mean_row[1]:>6} {mean_row[2]:>8.4f} {mean_row[3]:>8.4f} {mean_row[4]:>8.2f} "
        f"{mean_row[5]:>8.4f} {mean_row[6]:>8.4f} {mean_row[7]:>8.2f} "
        f"{mean_row[8]:>6} "
        f"{mean_row[9]:>8.4f} {mean_row[10]:>8.4f} {mean_row[11]:>8.2f} "
        f"{mean_row[12]:>6}"
    )

    summary_csv = os.path.join(OUT_BASE, "summary_5fold_Si5.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "n_rows", "Mean IOU", "Mean Dice", "Mean HD95",
            "(Organ) Mean IOU", "(Organ) Mean Dice", "(Organ) Mean HD95", "(Organ) n",
            "(Instr) Mean IOU", "(Instr) Mean Dice", "(Instr) Mean HD95", "(Instr) n",
        ])
        for r in rows_for_csv:
            writer.writerow(r)
        writer.writerow(mean_row)
    log.info(f"\nSummary saved to {summary_csv}")
    log.info("Done!")


if __name__ == "__main__":
    main()
