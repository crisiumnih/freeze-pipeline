import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import sys

# args
img_path = sys.argv[1]
out_path = sys.argv[2]

# load model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda().eval()

# preprocessing
transform = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# load image
img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0).cuda()

# forward pass
with torch.no_grad():
    feat = model(x)         # shape: [1, 1024] for ViT-L/14

feat = feat.squeeze().cpu().numpy()

# save
np.save(out_path, feat)

print("Saved:", out_path, "Shape:", feat.shape)
