# ✅ Weekly To-Do List (May 4–11)

## 🧑‍💻 Leen — *Sharpening & Evaluation Lead*

### 🔹 Day 1–2: Setup + AOD-Net
- [ ] Set up project environment (Python, dependencies).
- [ ] Download & resize UIEB images to 256×256.
- [ ] Clone AOD-Net repo & verify it runs on 1–2 images.
- [ ] Generate AOD-sharpened outputs (Group B).

### 🔹 Day 3–4: Folder Structure + Script Prep
- [ ] Organize output folders (`raw/`, `sharpened/`, etc.).
- [ ] Prepare PSNR & SSIM evaluation scripts (for later use on final outputs).

### 🔹 Day 5–6: Metric Evaluation (after Sümeyra finishes)
- [ ] Run PSNR & SSIM on:
  - Raw → FUnIE-GAN  
  - Raw → WaterGAN  
  - AOD → FUnIE-GAN  
  - AOD → WaterGAN
- [ ] Generate comparison plots (bar graphs, boxplots).
- [ ] Save results as CSV for report use.

### 🔹 Day 7: Final Touches
- [ ] Finalize code comments & cleanup.
- [ ] Write short summary of evaluation results.

---

## 🧑‍💻 Sümeyra — *Color Correction & Visuals Lead*

### 🔸 Day 1–2: Setup + Models
- [ ] Clone FUnIE-GAN and WaterGAN repos.
- [ ] Test both on 1–2 raw UIEB images.
- [ ] Confirm inference pipeline works.

### 🔸 Day 3–4: Batch Inference
- [ ] Run models on Group A (raw).
- [ ] After Leen shares Group B, run models on sharpened images.
- [ ] Save all outputs (4 versions per input image).

### 🔸 Day 5–6: Visual Output
- [ ] Select 3–5 examples and create side-by-side comparison grids:
  - Raw vs. AOD
  - FUnIE-GAN vs. AOD+FUnIE
  - WaterGAN vs. AOD+WaterGAN
- [ ] Save visuals in `results/examples/`.

### 🔸 Day 7: Final Touches
- [ ] Organize outputs and label clearly.
- [ ] Write bullet-point notes summarizing visual results.

---

## 🤝 Shared Tasks
- [ ] Sync outputs in shared folder (`data/`, `results/`).
- [ ] Finalize figures and tables for the report.
- [ ] Coordinate on what to include in the Results & Discussion section.
