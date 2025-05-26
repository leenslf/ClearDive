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
project_root = os.path.abspath("FUnIE-GAN_model")
sys.path.append(project_root)

from nets.funiegan import GeneratorFunieGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli doğrudan yükle
model = GeneratorFunieGAN().to(device)
model_path = os.path.join("FUnIE-GAN_model/models", "funie_generator.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Görsel dönüştürme işlemi
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Test klasörleri tanımı
test_cases = [
    ("input_images/100_dehazed_input_image", "ground_truth", "output/output_images_dehazed"),
    ("input_images/100_raw_input_image", "ground_truth", "output/output_images_raw")
]

# Her test durumu için PSNR / SSIM hesapla
for input_dir, gt_dir, output_dir in test_cases:
    os.makedirs(output_dir, exist_ok=True)
    total_psnr, total_ssim, count = 0, 0, 0

    for filename in os.listdir(input_dir):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue

        input_path = os.path.join(input_dir, filename)
        gt_path = os.path.join(gt_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if not os.path.exists(gt_path):
            print(f"❌ GT bulunamadı: {filename}")
            continue

        input_img = Image.open(input_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        input_tensor = transform(input_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        vutils.save_image(output_tensor, output_path)

        output_np = output_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        gt_np = transform(gt_img).numpy().transpose(1, 2, 0)

        output_np = (output_np * 255).astype(np.uint8)
        gt_np = (gt_np * 255).astype(np.uint8)

        total_psnr += psnr(gt_np, output_np, data_range=255)
        total_ssim += ssim(gt_np, output_np, channel_axis=2, data_range=255)
        count += 1

    if count > 0:
        print(f"\n✅ Test tamamlandı: {input_dir}")
        print(f"📌 Ortalama PSNR : {total_psnr / count:.2f} dB")
        print(f"📌 Ortalama SSIM : {total_ssim / count:.4f}")
    else:
        print(f"\n⚠️ Hiç eşleşen görüntü bulunamadı: {input_dir}")

