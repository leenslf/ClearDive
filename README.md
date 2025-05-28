# ClearDive

ClearDive is a modular research project exploring underwater image enhancement using both RGB and event data. Our pipeline combines model-based dehazing, GAN-based color correction, and edge-aware sharpening using event cameras. The project is structured around interpretable, lightweight stages to better analyze each component's contribution.

**Processed Images & Results:** [Google Drive Link](https://drive.google.com/drive/folders/1qMPggJ1J8m95xkhd0kFAGMM9vlvLGX4k?usp=sharing)

## 📌 Project Objectives

- Restore structural sharpness in underwater scenes with model-based dehazing.
- Apply GAN-driven color correction for visual fidelity.
- Explore event-guided techniques for recovering motion-blurred edges.
- Evaluate using both reference and no-reference image quality metrics.

## 🛠️ Pipeline Overview

1. **Input**: Raw RGB images from UIEB and EUVP datasets.
2. **Stage 1 – Dehazing**: Fine-tuned [`AOD-Net`](./AOD-Net-colab) for restoring structure.
3. **Stage 2 – Color Correction**: Using [`FUnIE-GAN`](./FUnIE-GAN) and [`UIEC2Net`](./UIEC2Net).
4. **Stage 3 – Event-Based Sharpening**: Event-guided fusion via [`davis-underwater-deblur`](./davis-underwater-deblur) using DAVIS data.
5. **Evaluation**: Metrics include PSNR, SSIM, UIQM, UCIQE, Laplacian variance, entropy, and edge density.

## 📂 Folder Structure

- `AOD-Net-colab/` – PyTorch AOD-Net fork fine-tuned on underwater datasets.
- `FUnIE-GAN/` – GAN-based color enhancement model.
- `UIEC2Net/` – CNN enhancement using dual color space features.
- `davis-underwater-deblur/` – Scripts and methods for event-RGB fusion.
- `todo.md` – Remaining tasks, work-in-progress notes.

## 📚 Datasets Used

- **UIEB** – 950 real-world underwater images, 890 with references.
- **EUVP** – 11k paired/unpaired underwater images from varied sources.
- **DAVIS-NUIUIED** – Synchronized underwater RGB and event streams.


