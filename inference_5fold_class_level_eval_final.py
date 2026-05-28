import json, os, sys, re, glob, time, csv
import numpy as np
import torch, yaml
from PIL import Image
from medpy import metric
from tqdm import tqdm
from collections import defaultdict
import logging

sys.path.insert(0, "/home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA/src")
sys.path.insert(0, "/home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA")
from lora import LoRA_sam
from processor import Samprocessor
from dataloader import DatasetSegmentation, collate_fn
from segment_anything import build_sam_vit_b
from torch.utils.data import DataLoader

SAM_CKPT = "/mnt/hdd2/task2/sam/sam_vit_b_01ec64.pth"
TEST_JSON = "/mnt/hdd2/task2/sam_lora/output_bbox_test.json"
GT_MASK_DIR = "/mnt/hdd2/task2/sam_lora/test/masks"
OUT_BASE = "/mnt/hdd2/task2/sam_lora/class_eval_5fold_final"
RANK = 2
DEVICE = "cuda:0"
ALL_CLASSES = list(range(1, 29))
ORGAN = {26, 27, 28}
INSTR = set(range(1, 26))
TEST_PATIENTS = ["19", "24", "71", "76", "78"]

FOLD_CKPT = {
    0: "/mnt/hdd2/task2/sam_lora/exp_3/lora_rank2_35_epoch_in_100_epochs_final_3.safetensors",
    1: "/mnt/hdd2/task2/sam_lora/exp_6/lora_rank2_15_epoch_in_100_epochs_final_6.safetensors",
    2: "/mnt/hdd2/task2/sam_lora/exp_5/lora_rank2_22_epoch_in_100_epochs_final_5.safetensors",
    3: "/mnt/hdd2/task2/sam_lora/exp_7/lora_rank2_24_epoch_in_100_epochs_final_7.safetensors",
    4: "/mnt/hdd2/task2/sam_lora/exp_8/lora_rank2_27_epoch_in_100_epochs_final_8.safetensors",
}

os.makedirs(OUT_BASE, exist_ok=True)

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[logging.FileHandler(os.path.join(OUT_BASE, "eval.log")),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def mean_valid(lst):
    v = [x for x in lst if not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def load_test_index():
    with open(TEST_JSON) as f:
        data = json.load(f)
    test_data = data.get("test", {})
    prompt_index = []
    for img_name in sorted(test_data.keys()):
        for info in test_data[img_name]:
            m = re.search(r"_class(\d+)\.png", info["mask_path"])
            cls_id = int(m.group(1)) if m else -1
            prompt_index.append((img_name, cls_id))
    return prompt_index, test_data


def run_fold_inference(fold, prompt_index):
    """Run inference for one fold. Returns (inst_results, pred_mask_dir)."""
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

    with open("/home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    processor = Samprocessor(model)
    dataset = DatasetSegmentation(config, processor, mode="test")
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    # Check resume
    existing_masks = set(os.listdir(pred_dir)) if os.path.exists(pred_dir) else set()
    total = len(dataloader)
    n_existing = len(existing_masks)
    if n_existing > 0:
        log.info(f"  Found {n_existing} existing pred masks, will skip already-saved")

    inst_results = []
    t_start = time.time()

    for i, batch in enumerate(tqdm(dataloader, desc=f"Fold{fold} Inference", total=total)):
        img_name, cls_id = prompt_index[i]
        pred_name = f"{img_name.replace('.png','')}_class{cls_id}.png"

        gt_mask = batch[0]["ground_truth_mask"].numpy().astype(bool)

        if pred_name in existing_masks:
            # Load saved pred
            pred_bin = np.array(Image.open(os.path.join(pred_dir, pred_name))) > 0
        else:
            with torch.no_grad():
                outputs = model(batched_input=batch, multimask_output=False)
            pred_mask = outputs[0]["masks"]
            pred_bin = (pred_mask > 0.5).cpu().numpy().squeeze().astype(np.uint8) * 255
            Image.fromarray(pred_bin).save(os.path.join(pred_dir, pred_name))
            pred_bin = pred_bin > 0

        inter = np.logical_and(pred_bin, gt_mask).sum()
        union = np.logical_or(pred_bin, gt_mask).sum()
        iou = inter / union if union > 0 else 0.0
        dice = 2 * inter / (pred_bin.sum() + gt_mask.sum() + 1e-8)
        try:
            hd95 = float(metric.binary.hd95(pred_bin, gt_mask)) if pred_bin.any() and gt_mask.any() else np.nan
        except Exception:
            hd95 = np.nan

        inst_results.append({"img": img_name, "class": cls_id, "iou": iou, "dice": dice, "hd95": hd95})

    t_elapsed = (time.time() - t_start) / 60
    log.info(f"  Fold {fold} inference: {t_elapsed:.1f} min, {len(inst_results)} instances")
    return inst_results, pred_dir


def run_class_eval(fold, pred_dir, test_data):
    """Compute class-level metrics for one fold's predictions."""
    log.info(f"Fold {fold}: computing class-level metrics...")
    t_start = time.time()

    patient_images = defaultdict(list)
    for img_name in sorted(test_data.keys()):
        pid = img_name.split("_")[0]
        if pid in TEST_PATIENTS:
            patient_images[pid].append(img_name)

    cls_results = []

    for pid in TEST_PATIENTS:
        images = patient_images[pid]
        for img_name in tqdm(images, desc=f"Fold{fold} Class {pid}", leave=False):
            base = img_name.replace('.png', '')
            pattern = f"{base}_class*.png"

            gt_map = np.zeros((1024, 1024), dtype=np.uint8)
            for f in sorted(glob.glob(os.path.join(GT_MASK_DIR, pattern))):
                cls = int(os.path.basename(f).replace('.png', '').rsplit('_class', 1)[1])
                gt_map[np.array(Image.open(f)) > 0] = cls

            pred_map = np.zeros((1024, 1024), dtype=np.uint8)
            for f in sorted(glob.glob(os.path.join(pred_dir, pattern))):
                cls = int(os.path.basename(f).replace('.png', '').rsplit('_class', 1)[1])
                pred_map[np.array(Image.open(f)) > 0] = cls

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
                    hd95 = float(metric.binary.hd95(pred_bin, gt_bin)) if pred_bin.any() and gt_bin.any() else np.nan
                except Exception:
                    hd95 = np.nan
                cls_results.append({"patient": pid, "img": img_name, "class": cls,
                                    "iou": iou, "dice": dice, "hd95": hd95})

    t_elapsed = (time.time() - t_start) / 60
    log.info(f"  Fold {fold} class eval: {t_elapsed:.1f} min, {len(cls_results)} per-class rows")
    return cls_results


def summarize(results_list, key_prefix=""):
    all_iou = [r["iou"] for r in results_list]
    all_dice = [r["dice"] for r in results_list]
    all_hd95 = [r["hd95"] for r in results_list if not np.isnan(r["hd95"])]
    org_hd95 = [r["hd95"] for r in results_list if r["class"] in ORGAN and not np.isnan(r["hd95"])]
    ins_hd95 = [r["hd95"] for r in results_list if r["class"] in INSTR and not np.isnan(r["hd95"])]
    return {
        "mIoU": np.mean(all_iou), "mDice": np.mean(all_dice),
        "mHD95": np.mean(all_hd95) if all_hd95 else float("nan"),
        "Organ HD95": np.mean(org_hd95) if org_hd95 else float("nan"),
        "Instr HD95": np.mean(ins_hd95) if ins_hd95 else float("nan"),
        "n": len(results_list),
    }

prompt_index, test_data = load_test_index()
log.info(f"5-Fold Class-Level Evaluation (FINAL weights). Total instances per fold: {len(prompt_index)}")
log.info(f"Folds: {list(FOLD_CKPT.keys())}")

all_fold_inst = {}
all_fold_cls = {}

for fold in range(5):
    log.info(f"\n{'='*60}")
    log.info(f"FOLD {fold}")
    log.info(f"{'='*60}")

    # Inference
    inst_results, pred_dir = run_fold_inference(fold, prompt_index)
    all_fold_inst[fold] = summarize(inst_results)

    # Class-level
    cls_results = run_class_eval(fold, pred_dir, test_data)
    all_fold_cls[fold] = summarize(cls_results)

    # Save fold CSV
    csv_path = os.path.join(OUT_BASE, f"fold{fold}", "class_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "img", "class", "iou", "dice", "hd95"])
        for r in cls_results:
            writer.writerow([r["patient"], r["img"], r["class"], r["iou"], r["dice"], r["hd95"]])

    log.info(f"  Fold {fold}: inst HD95={all_fold_inst[fold]['mHD95']:.2f}, class HD95={all_fold_cls[fold]['mHD95']:.2f}")

log.info(f"\n{'='*70}")
log.info("5-FOLD SUMMARY: SAM_LoRA Class-Level Evaluation")
log.info(f"{'='*70}")

log.info(f"{'Fold':<8} {'Inst HD95':>12} {'Class HD95':>12} {'Class mIoU':>12} {'Organ HD95':>12} {'Instr HD95':>12}")
avg_inst_hd95, avg_cls_hd95, avg_cls_iou, avg_org_hd95, avg_ins_hd95 = [], [], [], [], []
for fold in range(5):
    i = all_fold_inst[fold]
    c = all_fold_cls[fold]
    log.info(f"{fold:<8} {i['mHD95']:>12.2f} {c['mHD95']:>12.2f} {c['mIoU']:>12.4f} {c['Organ HD95']:>12.2f} {c['Instr HD95']:>12.2f}")
    avg_inst_hd95.append(i['mHD95'])
    avg_cls_hd95.append(c['mHD95'])
    avg_cls_iou.append(c['mIoU'])
    avg_org_hd95.append(c['Organ HD95'])
    avg_ins_hd95.append(c['Instr HD95'])

log.info(f"{'Mean':<8} {np.mean(avg_inst_hd95):>12.2f} {np.mean(avg_cls_hd95):>12.2f} {np.mean(avg_cls_iou):>12.4f} {np.mean(avg_org_hd95):>12.2f} {np.mean(avg_ins_hd95):>12.2f}")

# Save summary CSV
summary_csv = os.path.join(OUT_BASE, "summary_5fold_final.csv")
with open(summary_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["fold", "inst_HD95", "class_mIoU", "class_mDice", "class_mHD95", "class_Organ_HD95", "class_Instr_HD95"])
    for fold in range(5):
        i = all_fold_inst[fold]
        c = all_fold_cls[fold]
        writer.writerow([fold, i['mHD95'], c['mIoU'], c['mDice'], c['mHD95'], c['Organ HD95'], c['Instr HD95']])
    writer.writerow(["mean", np.mean(avg_inst_hd95), np.mean(avg_cls_iou), float('nan'), np.mean(avg_cls_hd95), np.mean(avg_org_hd95), np.mean(avg_ins_hd95)])

log.info(f"\nSummary saved to {summary_csv}")
log.info("Done!")
