"""Unified 5-fold aggregation for SAM-LoRA — pooled all-row averaging.

Same convention as nnUNet / Longiseg:
  Overall = mean over all rows whose class in {1..28} (drops class 29 'Other'
            so the prompt set matches nnUNet / Longiseg exactly: 23233 rows/fold).
  Organ   = mean over rows whose class in {26,27,28}
  Instr   = mean over rows whose class in {1..25}
"""
import os
import numpy as np
import pandas as pd

OUT_PATH = "/mnt/hdd2/task2/sam_lora/sam_lora_5fold.csv"
ORGAN = set([26, 27, 28])
INSTR = set(range(1, 26))
VALID = ORGAN | INSTR  # 1..28


def fold_csv(f):
    return f"/mnt/hdd2/task2/sam_lora/eval_{f}fold_test_final/results_inf_eval_rank2.csv"


def pooled(df, mask=None):
    sub = df if mask is None else df[mask]
    return (
        float(sub["IoU"].mean()),
        float(sub["Dice"].mean()),
        float(sub["HD95"].mean()),
        len(sub),
    )


def main():
    rows = []
    for f in range(5):
        df = pd.read_csv(fold_csv(f))
        df["HD95"] = df["HD95"].replace([np.inf, -np.inf], np.nan)
        df = df[df["class_id"].isin(VALID)].copy()
        ov = pooled(df)
        og = pooled(df, df["class_id"].isin(ORGAN))
        it = pooled(df, df["class_id"].isin(INSTR))
        rows.append({
            "fold": f,
            "n_rows": ov[3],
            "Mean IOU": ov[0],
            "Mean Dice": ov[1],
            "Mean HD95": ov[2],
            "(Organ) Mean IOU": og[0],
            "(Organ) Mean Dice": og[1],
            "(Organ) Mean HD95": og[2],
            "(Organ) n": og[3],
            "(Instr) Mean IOU": it[0],
            "(Instr) Mean Dice": it[1],
            "(Instr) Mean HD95": it[2],
            "(Instr) n": it[3],
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}\n")

    pd.set_option("display.float_format", lambda x: f"{x:.5f}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
