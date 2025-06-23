# project_964
Brain Tumor AI Model
# Brain-MRI Tumour Segmentation  
*Transfer-learning from BraTS 2020 to Figshare single-slice dataset*

---

## 1 · Project overview
This repo shows how a 3-D four-modal BraTS 2020 U-Net + ResNet-50[ Backbone ] encoder can be
re-used to segment **single T1-post slices** from the Figshare brain-tumour
collection (meningioma, glioma, pituitary).  
Key steps:

| step | what we do |
|------|------------|
| **01** | convert every Figshare `.mat` into `slice.png` + `slice_mask.png`; catalogue metadata in `master_split_png.csv` |
| **02** | build DataLoaders with Albumentations – resize 240², flips, elastic, Gaussian noise, brightness/contrast |
| **03** | patch BraTS checkpoint: first conv 4-ch → 1-ch (mean weights); copy all deeper layers |
| **04** | fine-tune with BCE + Dice (+ Focal after epoch 30); ReduceLROnPlateau scheduler |
| **05** | post-process: probability ≥ 0.30, keep largest connected component |
| **06** | evaluate on hold-out test set, plot GT vs pred, store Dice/volume CSV |
| **07** | provide a mini **query-bot** – one command returns tumour class, Dice, volume, centroid and an overlay PNG |

Result: **≈ 0.45 mean Dice** in 60 epochs (~ 5 GPU-hour) – a solid baseline
without training a new network from scratch.

---

## 2 · Datasets

| dataset | modality | slices | tumours |
|---------|----------|--------|---------|
| **BraTS 2020 (training)** | 4 × 3-D NIfTI (T1, T1ce, T2, FLAIR) | 369 patients | HGG, LGG |
| **Figshare tumour slices** | single T1-post PNG | 3 064 slices | 1 426 glioma, 930 pituitary, 708 meningioma |

*All Figshare PNGs are stored in* `figshare_sorted/<class>/####.png`  
*Master metadata:* `figshare_sorted/master_split_png.csv`

---

## 3 · Repo layout
.
├── demo_package/
│ ├── models/
│ │ ├── brats_best.pt ← 4-channel BraTS checkpoint
│ │ └── figshare_best.pt ← fine-tuned 1-channel model
│ └── scripts/
│ ├── unet_resnet50_1ch.py
│ └── mri_query_bot.py
│
├── notebooks/ ← Colab / Jupyter demos
├── train_figshare.py ← CLI fine-tune script
└── README.md


---

## 4 · Results

| metric (240 × 240)    | value |
| --------------------- | ----- |
| **Mean Dice (val)**   | 0.435 |
| **Mean Dice (test)**  | 0.448 |
| **Best slice Dice**   | 0.837 |
| **Median slice Dice** | 0.455 |

---

## 5 · Future Work

BraTS weights give instant stability; random init needs 10 × epochs.
Domain gap (3-D → 2-D) limits ceiling (~0.45 Dice).
Elastic + focal loss adds ~0.05 Dice.
Next steps: class-balanced sampler, CutMix, 2-D UNet++ ⇒ aim ≥ 0.60 Dice.

---

## 6 · Licence

Code – MIT
BraTS dataset – CC-BY-NC-SA
Figshare single-slice dataset – CC-BY-Cheng, Jun, et al(2015)

---

## 7 · Quick start (Colab) <a id="quickstart"></a>

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Clone this repo
!git clone https://github.com/<YOUR-GH-USER>/project_964.git
%cd project_964

# Install requirements
!pip install albumentations==1.4.3 torch torchvision tqdm scikit-image h5py

# Run the demo notebook (segmentation + query-bot)
!jupyter nbconvert --execute notebooks/Figshare_demo.ipynb

python train_figshare.py \
  --csv  "/content/drive/.../master_split_png.csv" \
  --root "/content/drive/.../figshare_sorted" \
  --brats-ckpt "/content/drive/.../brats_best.pt" \
  --out-dir models/figshare_run1 \
  --epochs 60 --batch 16

from demo_package.scripts.mri_query_bot import FigshareBot

png = "/content/drive/.../figshare_sorted/glioma/2195.png"
bot = FigshareBot()
print(bot.answer(png))

```

---



