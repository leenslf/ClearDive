import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils

# 1. Define the FUnIE-GAN_model folder and sys.add Oct to path
funiegan_root = os.path.abspath("FUnIE-GAN_model")
sys.path.append(funiegan_root)

# 2. Import the model (from the nets)
from nets.funiegan import GeneratorFunieGAN

# 3. Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. Create and upload the model
model = GeneratorFunieGAN().to(device)
model_path = os.path.join(funiegan_root, "models", "funie_generator.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 5. Upload the entrance image
input_path = os.path.join("input_images", "input2_for_inference.jpg")
img = Image.open(input_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

input_tensor = transform(img).unsqueeze(0).to(device)

# 6. Pass it through the model
with torch.no_grad():
    output_tensor = model(input_tensor)

# 7. Save the output
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "output2_inference_funiegan.jpg")
vutils.save_image(output_tensor, output_path)

print(f"🎉 The image was successfully improved and saved here: {output_path}")

 