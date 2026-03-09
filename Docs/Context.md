# Project Context: Alzheimer's Detection via 3D MRI (TFG)

## 1. Project Definition
**Goal:** Develop a Deep Learning pipeline to classify 3D Structural MRI brain scans into three stages of Alzheimer's Disease.
**Type:** Undergraduate Thesis (TFG).
**Core Philosophy:** "Low-Resource & Efficient". The solution must run on consumer-grade GPUs (Google Colab / Local GPU). We prioritize clean code and methodological rigor over SOTA accuracy.

## 2. Technical Stack & Constraints
* **Language:** Python 3.9+
* **Deep Learning Framework:** PyTorch.
* **Medical Framework:** MONAI (Mandatory for transforms and loading).
* **Data Handling:** Nibabel, Pandas, NumPy.
* **Visualization:** Matplotlib, Streamlit (for final demo).
* **Hardware Constraint:** **High VRAM restriction**. Input volumes must be downsampled (e.g., 96x96x96) to avoid OOM.
* **Repo Structure:** Modular (separation of `src/` and `notebooks/`).

## 3. Data Strategy
### Primary Dataset: OASIS-1 (Cross-Sectional)
* **Format:** ANALYZE 7.5 pairs (`.img` / `.hdr`). Treated as NIfTI.
* **Specific Files:** `*_masked_gfc.img` (Skull-stripped, Gain Field Corrected, Atlas Registered). **Do not use Raw or Freesurfer data.**
* **Input Shape:** 3D Volumetric Data (Single channel).
* **Classes (Labels):** Based on `CDR` (Clinical Dementia Rating) from clinical CSV.
    * **Class 0 (CN - Control):** CDR = 0
    * **Class 1 (MCI - Mild):** CDR = 0.5
    * **Class 2 (AD - Alzheimer):** CDR ≥ 1
* **Splitting:**
    * Train (70%) / Val (15%) / Test (15%).
    * **Must use Stratified Split** based on Label to maintain class balance.
    * Splits must be fixed in saved CSV files (`data/splits/`), not random every run.

## 4. Pipeline Architecture
### A. Preprocessing (On-the-fly via MONAI)
1.  **LoadImaged:** Load `.img` files.
2.  **EnsureChannelFirstd:** Shape becomes `(Channel, D, H, W)`.
3.  **Orientationd:** Force 'RAS' orientation.
4.  **ScaleIntensityRangePercentilesd:** Normalize intensities (clip outliers 1-99%).
5.  **Resized:** Downsample to `(96, 96, 96)` for memory efficiency.

### B. Augmentation (Train only)
* `RandRotate90d`, `RandFlipd` (prob=0.1), `RandGaussianNoised`.

### C. Model
* **Baseline:** Custom `Simple3DCNN` (4 layers of Conv3d + BatchNorm + ReLU + MaxPool).
* **Advanced:** `DenseNet121` (3D version) or `EfficientNet-3D` (only if Baseline underperforms).
* **Loss:** `CrossEntropyLoss` (with class weights if imbalance is severe).
* **Optimizer:** Adam.

### D. Explainability (XAI)
* **Grad-CAM:** Required for the final deliverable. Must visualize activation maps on the central slice of the MRI.

## 5. Implementation Roadmap (Checkpoints)

### Phase 1: Data Engineering (Current Priority)
* [ ] Script to extract `masked_gfc` files from OASIS tarballs to `data/raw/images`.
* [ ] Script to parse `oasis_cross-sectional.csv` and map CDR to classes 0, 1, 2.
* [ ] Generate `train.csv`, `val.csv`, `test.csv`.

### Phase 2: Pipeline Verification
* [ ] Implement MONAI `Dataset` and `DataLoader`.
* [ ] **Sanity Check:** Visualize a batch of images after preprocessing (verify orientation and skull-stripping).

### Phase 3: Modeling
* [ ] Implement `Simple3DCNN`.
* [ ] **"Overfit Batch" Test:** Train on 4 images until Loss ~ 0.
* [ ] Full training loop with Validation metrics.

### Phase 4: Refinement
* [ ] Add Data Augmentation.
* [ ] Generate Confusion Matrix & F1-Score.
* [ ] Implement Grad-CAM visualization.

## 6. Rules for AI Agents
1.  **No Data Leakage:** Never mix subjects between Train and Test splits.
2.  **No Magic Numbers:** Use `src/config.py` for dimensions, paths, and hyperparameters.
3.  **Memory First:** Always assume limited VRAM. Don't suggest large batch sizes (use 2, 4, or 8).
4.  **Medical Validity:** Do not distort the aspect ratio of the brain. Use isotropic resizing where possible.
5.  **No Synthetic Data:** Do not suggest GANs/Diffusion for data generation at this stage.