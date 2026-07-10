"""
Fig.2 定性图: 连续帧上的时序一致性对比
Rows: Raw / Ground Truth / w/o LRU (SAM-LoRA) / Ours (SAM-LRU)
- 底部类别图例(仅列该片段 GT 中出现的类)
- 输出 png / pdf / svg
单进程顺序执行, 逐帧打印真实 IoU + overlay 像素数自检(避免并行输出伪影导致的假象)。

依赖: numpy, pillow, matplotlib
运行: python3 plot_fig2_qualitative.py
"""

import numpy as np, glob, re, os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

os.chdir('/mnt/hdd2/task2')

P = '19_C1'
frames = ['18825', '18967', '19110', '19252', '19349', '19446']
OUT = '/home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/Sam_LoRA/fig2_oneRow'

RAW  = 'sam_lora/test/images'                              # {P}-{fr}.png
GT   = 'sam_lora/test/masks'                               # {P}-{fr}_class{N}.png (0/255)
LORA = f'sam_lora/plots_{P.split("_")[0]}'                 # {P}-{fr}_class{N}_rank2.png
LRU  = 'sam_lora_lru/result_final/predict_masks/fold_0'    # {P}-{fr}.png (index map)

CM = [
    (1,[.2,1,1]),(2,[0,.2,.8]),(3,[.2,1,1]),(4,[0,.2,.8]),(5,[.2,1,1]),(6,[0,.2,.8]),
    (7,[.2,1,1]),(8,[0,.2,.8]),(9,[.2,1,1]),(10,[0,1,0]),(11,[0,.2,.8]),(12,[.2,1,1]),
    (13,[0,1,0]),(14,[0,.2,.8]),(15,[.2,1,1]),(16,[0,1,0]),(17,[0,.2,.8]),(18,[.2,1,1]),
    (19,[0,.2,.8]),(20,[.2,.6,1]),(21,[1,1,0]),(22,[1,.8,0]),(23,[.2,.2,.4]),(24,[.8,.2,.8]),
    (25,[1,.8,.8]),(26,[.8,0,0]),(27,[1,.4,.4]),(28,[.3333,0,.498])
]

pal = np.zeros((29, 3), np.uint8)

for v, c in CM:
    pal[v] = (np.array(c) * 255).astype(np.uint8)

LEG = [
    ('Tool Clasper', [.2,1,1]),
    ('Tool Shaft', [0,.2,.8]),
    ('Tool Wrist', [0,1,0]),
    ('Suturing Needle', [1,1,0]),
    ('Thread', [1,.8,0]),
    ('Trocar', [1,.8,.8]),
    ('Hem-o-lock', [.8,.2,.8]),
    ('Renal Wound', [.8,0,0]),
    ('Kidney', [1,.4,.4]),
    ('Resected Tumor', [.3333,0,.498])
]

sem = {
    1:0, 2:1, 3:0, 4:1, 5:0, 6:1, 7:0, 8:1, 9:0, 10:2, 11:1, 12:0, 13:2, 14:1,
    15:0, 16:2, 17:1, 18:0, 19:1, 21:3, 22:4, 25:5, 24:6, 26:7, 27:8, 28:9
}


def idx_bin(d, fr, suf):
    idx = np.zeros((1024, 1024), np.uint8)

    for f in sorted(glob.glob(f'{d}/{P}-{fr}_class*{suf}.png')):
        c = int(re.search(r'class(\d+)', f).group(1))
        idx[np.array(Image.open(f).convert('L')) > 127] = c

    return idx


def ov(raw, idx, a=.55):
    o = raw.astype(float).copy()

    for c in np.unique(idx):
        if c == 0:
            continue

        m = idx == c
        o[m] = (1 - a) * o[m] + a * pal[c]

    return o.clip(0, 255).astype(np.uint8)


def miou(pred, gt):
    cs = np.unique(gt)
    cs = cs[cs > 0]

    v = [
        ((pred == c) & (gt == c)).sum() / max(((pred == c) | (gt == c)).sum(), 1)
        for c in cs
    ]

    return float(np.mean(v)) if len(v) else 0.0


# 黑边裁剪，用首帧检测底部黑边
g = np.array(Image.open(f'{RAW}/{P}-{frames[0]}.png').convert('L'))
nz = np.where(g.max(1) > 12)[0]

r0, r1 = int(nz[0]), int(nz[-1]) + 1
W = r1 - r0

present, il, ir, data = set(), [], [], []

for fr in frames:
    raw  = np.array(Image.open(f'{RAW}/{P}-{fr}.png').convert('RGB'))
    gt   = idx_bin(GT, fr, '')
    lora = idx_bin(LORA, fr, '_rank2')
    lru  = np.array(Image.open(f'{LRU}/{P}-{fr}.png'))

    present |= set(np.unique(gt)) - {0}

    a, b = miou(lora, gt), miou(lru, gt)
    il.append(a)
    ir.append(b)

    data.append((fr, raw, gt, lora, lru))

    chg_lora = (ov(raw, lora) != raw).any(2).sum()

    print(
        f'frame {fr}: IoU lora={a:.2f} lru={b:.2f} | '
        f'lora_overlay_px={chg_lora} | '
        f'lora_cls={sorted(set(np.unique(lora)) - {0})}'
    )


rows = ['Input Frame', 'Ground Truth', 'SAM-LoRA', 'SAM-LRU (Ours)']

fig, axes = plt.subplots(
    4,
    len(frames),
    figsize=(len(frames) * 2.6, 4 * 2.6 * W / 1024 + 0.9)
)

for j, (fr, raw, gt, lora, lru) in enumerate(data):
    imgs = [
        raw,
        ov(raw, gt),
        ov(raw, lora),
        ov(raw, lru)
    ]

    for i, im in enumerate(imgs):
        ax = axes[i, j]
        ax.imshow(im[r0:r1])
        ax.set_xticks([])
        ax.set_yticks([])

        if j == 0:
            ax.set_ylabel(
                rows[i],
                fontsize=14,
                fontweight='bold',
                labelpad=8
            )

        if i == 0:
            ax.set_title(
                f'Frame {j + 1}',
                fontsize=13,
                fontweight='bold',
                pad=8
            )


used = sorted({sem[c] for c in present if c in sem})
handles = [mpatches.Patch(color=LEG[k][1], label=LEG[k][0]) for k in used]

fig.legend(
    handles=handles,
    loc='lower center',
    ncol=max(len(handles), 1),
    fontsize=13,
    frameon=False,
    bbox_to_anchor=(0.5, 0.035),
    columnspacing=0.8,
    handlelength=1.1,
    handletextpad=0.35
)

plt.tight_layout(rect=[0, 0.06, 1, 1])

for e in ['png', 'pdf', 'svg']:
    plt.savefig(f'{OUT}.{e}', dpi=150, bbox_inches='tight')

print('legend:', [LEG[k][0] for k in used])
print('DONE ->', OUT + '.{png,pdf,svg}')