import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# 🔧 Yol ayarları
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_root = os.path.abspath("UIEC2Net_model")
sys.path.append(project_root)
# Model importu (core klasörü üstten tanıtılmış olmalı)
from core.Models.UWModels.UIEC2Net import UIEC2Net



def test(input_dir, gt_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # 🎯 Görsel dönüştürme
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # 🧠 Model yükle
    model = UIEC2Net().to(device)
    checkpoint = torch.load("UIEC2Net_model/checkpoints/UIEC2Net.pth", map_location=device)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    #model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    # 📊 Metrik toplamaları
    total_psnr, total_ssim, count = 0, 0, 0

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            gt_path = os.path.join(gt_dir, filename)

            if not os.path.exists(gt_path):
                print(f"❌ GT bulunamadı: {filename}")
                continue

            # Görselleri yükle
            input_img = Image.open(input_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")

            input_tensor = transform(input_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output_tensor = model(input_tensor)

            # Çıktıyı kaydet
            output_path = os.path.join(output_dir, filename)
            vutils.save_image(output_tensor, output_path)

            # PSNR ve SSIM hesapla
            output_np = output_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            #gt_np = transform(gt_img).numpy().transpose(1, 2, 0)
            gt_img = gt_img.resize((256, 256))  # ✅ GT ile model output aynı boyutta olsun
            gt_tensor = transform(gt_img)
            gt_np = gt_tensor.numpy().transpose(1, 2, 0)

            output_np = (output_np * 255).astype(np.uint8)
            gt_np = (gt_np * 255).astype(np.uint8)

            total_psnr += psnr(gt_np, output_np, data_range=255)
            total_ssim += ssim(gt_np, output_np, channel_axis=2, data_range=255)
            count += 1

    # 📢 Sonuçları yazdır
    if count > 0:
        print(f"\n✅ Test tamamlandı ({count} görüntü)")
        print(f"📌 Ortalama PSNR : {total_psnr / count:.2f} dB")
        print(f"📌 Ortalama SSIM : {total_ssim / count:.4f}")
    else:
        print("❌ Hiç eşleşen görsel bulunamadı.")

test("input_data/100_dehazed_input_image", "ground_truth", "output/output_images_dehazed")
test("input_data/100_raw_input_image", "ground_truth", "output/output_images_raw")