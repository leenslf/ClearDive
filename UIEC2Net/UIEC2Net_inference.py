import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils

# Uwenhancing-get the full path of the master folder (also happens relatively)
project_root = os.path.abspath("UIEC2Net_model")
sys.path.append(project_root)

# Now the model can be imported
from core.Models.UWModels.UIEC2Net import UIEC2Net
    

# Select the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create and upload the model
model = UIEC2Net().to(device)

# Own.update the path to your pth file
weight_path = r"UIEC2Net_model\checkpoints\UIEC2Net.pth"
checkpoint = torch.load(weight_path, map_location=device)
state_dict = checkpoint["state_dict"]
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

model.eval()

# Upload the image
input_path = r"input_data\input1_for_inference.jpg"  # the underwater image you want to test
img = Image.open(input_path).convert("RGB")

# Convert the image according to the model
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # according to the original model, it may be required
    transforms.ToTensor()
])

input_tensor = transform(img).unsqueeze(0).to(device)

# Put it through the model
with torch.no_grad():
    output_tensor = model(input_tensor)

# save the output
output_path = r"output\output1_uiec2net.jpg"
vutils.save_image(output_tensor, output_path)

print(f"The image was successfully improved and saved here: {output_path}")

