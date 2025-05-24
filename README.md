# ClearDive
ClearDive is a research project that explores a two-stage pipeline for enhancing underwater images. Stage 1 sharpens images using [AOD-Net](https://github.com/leenslf/AOD-Net-colab), while Stage 2 corrects colors with FUnIE-GAN and WaterGAN.We're also preparing to integrate event camera data for edge-aware sharpening in future iterations.

**Check out processed images:** [Drive](https://drive.google.com/drive/folders/1qMPggJ1J8m95xkhd0kFAGMM9vlvLGX4k?usp=sharing)


## 📌 Project Goals

Restore structural detail in underwater images using deep learning.

Apply GAN-based color correction for improved visual fidelity.

Evaluate how sharpening affects enhancement quality.

Explore RGB + event data fusion for improved edge reconstruction.

## 🛠️ Pipeline Overview

**Input**: Raw underwater images (UIEB dataset)

**Sharpening**: AOD-Net (pretrained)

**Color Correction**: FUnIE-GAN and UIEC^2-Net

**Evaluation**: PSNR, SSIM, UIQM, UCIQE

**(Future)**: Event-guided sharpening using DAVIS-NUIUIED

## 📂 Folder Structure


