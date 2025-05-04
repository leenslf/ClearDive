# ClearDive
ClearDive is a research project that explores a two-stage pipeline for enhancing underwater images.Stage 1 sharpens images using AOD-Net, while Stage 2 corrects colors with FUnIE-GAN and WaterGAN.We're also preparing to integrate event camera data for edge-aware sharpening in future iterations.

## 📌 Project Goals

Restore structural detail in underwater images using deep learning.

Apply GAN-based color correction for improved visual fidelity.

Evaluate how sharpening affects enhancement quality.

Explore RGB + event data fusion for improved edge reconstruction.

## 🛠️ Pipeline Overview

**Input**: Raw underwater images (UIEB dataset)

**Sharpening**: AOD-Net (pretrained)

**Color Correction**: FUnIE-GAN and WaterGAN

**Evaluation**: PSNR, SSIM, UIQM, UCIQE

**(Future)**: Event-guided sharpening using DAVIS-NUIUIED

## 📂 Folder Structure


