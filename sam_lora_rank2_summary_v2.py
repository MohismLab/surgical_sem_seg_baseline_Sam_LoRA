import os
import json
import pandas as pd
import numpy as np
import logging
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_class_id(mask_path):
    """从 mask_path 中提取类别 ID，例如 _class15.png -> 15"""
    try:
        match = re.search(r'_class(\d+)\.png', mask_path)
        if match:
            return int(match.group(1))
    except Exception as e:
        logger.error(f"Failed to extract class from {mask_path}: {e}")
    return -1

def get_flattened_mapping(json_path):
    """
    解析 JSON 映射文件。
    结构为 {"test": {"image_name": [{"mask_path": "...", "bbox": [...]}, ...], ...}}
    返回展平后的 class_id 列表，顺序严格遵循推理时的顺序（通常按文件名排序）。
    """
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    test_data = data.get('test', {})
    if not test_data:
        return None
    
    # 按照文件名排序键值，并展平每个图片的 mask 列表
    flattened_classes = []
    for img_name in sorted(test_data.keys()):
        for mask_item in test_data[img_name]:
            class_id = extract_class_id(mask_item['mask_path'])
            flattened_classes.append(class_id)
            
    return flattened_classes

def process_sam_lora_batch():
    base_dir = "/mnt/hdd2/task2/sam_lora"
    patients = ["19", "24", "71", "76", "78"]
    folds = ["0", "1", "2", "3", "4"]
    
    instrument_classes = set(range(1, 26))
    organ_classes = {26, 27, 28}
    
    for patient in patients:
        json_path = os.path.join(base_dir, f"output_bbox_test_{patient}.json")
        logger.info(f"Loading mapping for Patient {patient}...")
        mapping = get_flattened_mapping(json_path)
        
        if mapping is None:
            logger.warning(f"No valid mapping found for patient {patient}")
            continue
            
        for fold in folds:
            fold_folder = f"eval_{fold}fold_{patient}_final"
            csv_path = os.path.join(base_dir, fold_folder, "results_inf_eval_rank2.csv")
            output_path = os.path.join(base_dir, fold_folder, "result_organ_instrument.csv")
            
            if not os.path.exists(csv_path):
                continue
            
            try:
                # 读取 CSV，处理 Windows 换行符
                df = pd.read_csv(csv_path)
                
                # 筛选 Rank 2 模型
                # 注意：Model 列可能包含 "Rank 2"（带空格）
                df_rank2 = df[df['Model'].str.contains('Rank 2', case=False, na=False)].copy()
                
                if df_rank2.empty:
                    logger.warning(f"No 'Rank 2' data in {fold_folder}")
                    continue
                
                # 核心映射：Sample_Index -> mapping[Sample_Index]
                def map_idx_to_class(row):
                    idx = int(row['Sample_Index'])
                    if 0 <= idx < len(mapping):
                        return mapping[idx]
                    return -1
                
                df_rank2['class_id'] = df_rank2.apply(map_idx_to_class, axis=1)
                
                # 分组计算
                summary_rows = []
                for label, target_classes in [("Instrument", instrument_classes), ("Organ", organ_classes)]:
                    sub_df = df_rank2[df_rank2['class_id'].isin(target_classes)]
                    if not sub_df.empty:
                        summary_rows.append({
                            "Group": label,
                            "Mean Dice": sub_df['Dice'].mean(),
                            "Mean IoU": sub_df['IoU'].mean(),
                            "Mean HD95": sub_df['HD95'].replace([np.inf, -np.inf], np.nan).mean()
                        })
                
                if summary_rows:
                    pd.DataFrame(summary_rows).to_csv(output_path, index=False)
                    logger.info(f"Generated: {output_path}")
                else:
                    logger.warning(f"No categorized rows found in {fold_folder}")
                    
            except Exception as e:
                logger.error(f"Failed to process {csv_path}: {e}")

if __name__ == "__main__":
    process_sam_lora_batch()
    logger.info("All SAM_LoRA folders processed.")
