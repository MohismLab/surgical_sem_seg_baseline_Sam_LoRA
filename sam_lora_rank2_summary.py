import os
import json
import pandas as pd
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_sam_lora_metrics():
    base_dir = "/mnt/hdd2/task2/sam_lora"
    patients = ["19", "24", "71", "76", "78"]
    folds = ["0", "1", "2", "3", "4"]
    
    instrument_classes = set(range(1, 26))
    organ_classes = {26, 27, 28}
    
    for patient in patients:
        # 加载索引映射
        json_path = os.path.join(base_dir, f"output_bbox_test_{patient}.json")
        if not os.path.exists(json_path):
            logger.warning(f"JSON mapping not found: {json_path}")
            continue
            
        with open(json_path, 'r') as f:
            mapping = json.load(f)
        
        for fold in folds:
            fold_folder = f"eval_{fold}fold_{patient}_final"
            csv_path = os.path.join(base_dir, fold_folder, "results_inf_eval_rank2.csv")
            output_path = os.path.join(base_dir, fold_folder, "result_organ_instrument.csv")
            
            if not os.path.exists(csv_path):
                # 尝试其他可能的命名风格，有的可能没有 _final 或者 fold 序号不同
                logger.debug(f"CSV not found at {csv_path}, skipping fold {fold} for patient {patient}")
                continue
                
            try:
                df = pd.read_csv(csv_path)
                
                # 筛选 Rank 2 模型 (使用不区分大小写的包含匹配)
                # 用户提到 model 选为 rank2
                mask = df['Model'].str.contains('rank2', case=False, na=False) | df['Model'].str.contains('Rank 2', case=False, na=False)
                df_rank2 = df[mask].copy()
                
                if df_rank2.empty:
                    logger.warning(f"No 'Rank 2' data found in {csv_path}")
                    continue
                
                # 映射 ClassID
                # Sample_Index 是 JSON 中的索引
                def get_class(idx):
                    try:
                        idx = int(idx)
                        if 0 <= idx < len(mapping):
                            return mapping[idx]['class_id']
                    except:
                        pass
                    return -1
                
                df_rank2['class_id'] = df_rank2['Sample_Index'].apply(get_class)
                
                # 分组计算
                summary_rows = []
                for label, classes in [("Instrument", instrument_classes), ("Organ", organ_classes)]:
                    sub_df = df_rank2[df_rank2['class_id'].isin(classes)]
                    if not sub_df.empty:
                        # 确保数值为 float 且处理 NaN
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
                    logger.warning(f"No valid classes found for summary in {csv_path}")
                    
            except Exception as e:
                logger.error(f"Error processing {csv_path}: {e}")

if __name__ == "__main__":
    process_sam_lora_metrics()
    logger.info("Batch processing completed.")
