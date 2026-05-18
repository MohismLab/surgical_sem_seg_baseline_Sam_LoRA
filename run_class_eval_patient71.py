import json, os, sys, re, glob
import numpy as np
import torch
import yaml
from PIL import Image
from medpy import metric
from tqdm import tqdm
from collections import defaultdict

# Add SAM_LoRA to path
sys.path.insert(0, "/home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA/src")
sys.path.insert(0, "/home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA")

from lora import LoRA_sam
from processor import Samprocessor
from dataloader import DatasetSegmentation, collate_fn
from segment_anything import build_sam_vit_b
from torch.utils.data import DataLoader

# --- Config ---
CKPT = "/mnt/hdd2/task2/sam_lora/exp_3/lora_rank2_35_epoch_in_100_epochs_final_3.safetensors"
SAM_CKPT = "/mnt/hdd2/task2/sam/sam_vit_b_01ec64.pth"
TEST_JSON = "/mnt/hdd2/task2/sam_lora/output_bbox_test.json"
TEST_IMG_DIR = "/mnt/hdd2/task2/sam_lora/test/images"
GT_MASK_DIR = "/mnt/hdd2/task2/sam_lora/test/masks"
OUT_DIR = "/mnt/hdd2/task2/sam_lora/class_eval_demo"
PATIENT = "71"
RANK = 2
DEVICE = "cuda:0"
ALL_CLASSES = list(range(1, 29))
ORGAN = {26, 27, 28}
INSTR = set(range(1, 26))

os.makedirs(os.path.join(OUT_DIR, "pred_masks"), exist_ok=True)

# --- Load model ---
print("Loading SAM model...")
sam = build_sam_vit_b(checkpoint=SAM_CKPT)
sam_lora = LoRA_sam(sam, RANK)
sam_lora.load_lora_parameters(CKPT)
model = sam_lora.sam
model.eval()
model.to(DEVICE)

with open("./config.yaml") as f:
    config = yaml.load(f, Loader=yaml.Loader)

processor = Samprocessor(model)

# --- Load test JSON, filter to patient 71 ---
with open(TEST_JSON) as f:
    data = json.load(f)
test_data = data.get("test", {})

# Filter images for patient 71
patient_images = [n for n in sorted(test_data.keys()) if n.startswith(f"{PATIENT}_")]
print(f"Patient {PATIENT}: {len(patient_images)} images")

# --- Run inference ---
all_results = []  # per-instance results

for img_name in tqdm(patient_images, desc=f"Patient {PATIENT}"):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    image = Image.open(img_path)
    original_size = tuple(image.size)[::-1]  # (H, W)

    for info in test_data[img_name]:
        bbox = info["bbox"]
        cls_id = int(re.search(r"_class(\d+)\.png", info["mask_path"]).group(1))
        gt_mask = np.array(Image.open(info["mask_path"]).convert("1")).astype(bool)

        # Preprocess
        image_tensor = processor.process_image(image, original_size).to(DEVICE).float()
        boxes_tensor = processor.process_prompt(bbox, original_size).to(DEVICE)

        with torch.no_grad():
            outputs = model(
                batched_input=[{"image": image_tensor[0], "original_size": original_size,
                                "boxes": boxes_tensor[0]}],
                multimask_output=False,
            )
        pred_mask = outputs[0]["masks"]  # (1, H, W) at original_size

        # Save prediction mask
        pred_bin = (pred_mask > 0.5).cpu().numpy().squeeze().astype(np.uint8) * 255
        pred_save_name = f"{img_name.replace('.png','')}_class{cls_id}.png"
        Image.fromarray(pred_bin).save(os.path.join(OUT_DIR, "pred_masks", pred_save_name))

        # Instance-level metrics
        pred_bool = pred_bin > 0
        gt_bool = gt_mask
        inter = np.logical_and(pred_bool, gt_bool).sum()
        union = np.logical_or(pred_bool, gt_bool).sum()
        iou_inst = inter / union if union > 0 else 0.0
        dice_inst = 2 * inter / (pred_bool.sum() + gt_bool.sum() + 1e-8)
        try:
            hd95_inst = float(metric.binary.hd95(pred_bool, gt_bool)) if pred_bool.any() and gt_bool.any() else np.nan
        except Exception:
            hd95_inst = np.nan

        all_results.append({
            "img_name": img_name, "class": cls_id,
            "iou_inst": iou_inst, "dice_inst": dice_inst, "hd95_inst": hd95_inst,
        })

print(f"Inference done. {len(all_results)} instances processed.")

# ============================================================
# Class-level evaluation
# ============================================================
print("\n=== Class-Level Evaluation ===")

# Build per-image class maps
image_class_results = defaultdict(lambda: {"gt": {}, "pred": {}})

for img_name in patient_images:
    # Build GT class map
    gt_map = np.zeros((1024, 1024), dtype=np.uint8)
    for f in sorted(glob.glob(os.path.join(GT_MASK_DIR, f"{img_name.replace('.png','')}_class*.png"))):
        cls = int(os.path.basename(f).replace('.png', '').rsplit('_class', 1)[1])
        mask = np.array(Image.open(f)) > 0
        gt_map[mask] = cls

    # Build Pred class map
    pred_map = np.zeros((1024, 1024), dtype=np.uint8)
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "pred_masks", f"{img_name.replace('.png','')}_class*.png"))):
        cls = int(os.path.basename(f).replace('.png', '').rsplit('_class', 1)[1])
        mask = np.array(Image.open(f)) > 0
        pred_map[mask] = cls

    # Per-class metrics
    for cls in ALL_CLASSES:
        gt_bin = (gt_map == cls)
        pred_bin = (pred_map == cls)
        if not gt_bin.any():
            continue
        inter = (pred_bin & gt_bin).sum()
        union = (pred_bin | gt_bin).sum()
        iou_cls = inter / union if union > 0 else 0.0
        dice_cls = 2 * inter / (pred_bin.sum() + gt_bin.sum() + 1e-8)
        try:
            hd95_cls = float(metric.binary.hd95(pred_bin, gt_bin)) if pred_bin.any() and gt_bin.any() else np.nan
        except Exception:
            hd95_cls = np.nan
        image_class_results[img_name][cls] = {"iou": iou_cls, "dice": dice_cls, "hd95": hd95_cls}

# ============================================================
# Aggregate
# ============================================================
def summarize(name, values_list, key):
    vals = [r[key] for r in values_list if not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")

inst_ious = all_results
inst_org = [r for r in all_results if r["class"] in ORGAN]
inst_ins = [r for r in all_results if r["class"] in INSTR]

# Collect all class-level results
cls_results = []
for img_name, cls_dict in image_class_results.items():
    for cls, m in cls_dict.items():
        cls_results.append({"class": cls, **m})

cls_org = [r for r in cls_results if r["class"] in ORGAN]
cls_ins = [r for r in cls_results if r["class"] in INSTR]

print(f"\n{'='*70}")
print(f"Patient {PATIENT}: Instance-Level vs Class-Level")
print(f"{'='*70}")
print(f"{'':<25} {'Inst-Level':>18} {'Class-Level':>18} {'Gap':>10}")
print(f"{'Mean IoU':<25} {summarize('',inst_ious,'iou_inst'):>18.4f} {summarize('',cls_results,'iou'):>18.4f} {summarize('',inst_ious,'iou_inst')-summarize('',cls_results,'iou'):>10.4f}")
print(f"{'Mean Dice':<25} {summarize('',inst_ious,'dice_inst'):>18.4f} {summarize('',cls_results,'dice'):>18.4f} {summarize('',inst_ious,'dice_inst')-summarize('',cls_results,'dice'):>10.4f}")
print(f"{'Mean HD95':<25} {summarize('',inst_ious,'hd95_inst'):>18.2f} {summarize('',cls_results,'hd95'):>18.2f} {summarize('',inst_ious,'hd95_inst')-summarize('',cls_results,'hd95'):>10.2f}")
print()
print(f"  Organ  HD95:  inst={summarize('',inst_org,'hd95_inst'):.2f}  →  class={summarize('',cls_org,'hd95'):.2f}")
print(f"  Instr  HD95:  inst={summarize('',inst_ins,'hd95_inst'):.2f}  →  class={summarize('',cls_ins,'hd95'):.2f}")
print()

# Also show comparison with existing fold0 results for this patient
print(f"=== Cross-reference ===")
print(f"SAM_LoRA fold0 (existing, patient {PATIENT}): mIoU=0.82638, mHD95=7.23")
print(f"  → This is INSTANCE-LEVEL from eval_0fold_test_final/eval.log")
print(f"  → Our run should produce similar instance-level numbers")
