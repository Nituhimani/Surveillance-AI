#to test re_id model
#python test_reid.py path/to/reid_checkpoint.pth sample_a.jpg sample_b.jpg

import sys
from re_id import DNN_fixed, load_checkpoint_into_model, IMAGE_HEIGHT, IMAGE_WIDTH, device_select
from PIL import Image
import cv2
from torchvision import transforms
import torch

def to_tensor_from_bgr(bgr):
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    t = transforms.Compose([transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), transforms.ToTensor()])(pil).unsqueeze(0)
    return t

ckpt = sys.argv[1]
img1 = sys.argv[2]
img2 = sys.argv[3]

device = device_select(None)
model = DNN_fixed().to(device)
model = load_checkpoint_into_model(model, ckpt, map_location=device)
model.eval()

i1 = cv2.imread(img1)
i2 = cv2.imread(img2)
t1 = to_tensor_from_bgr(i1).to(device)
t2 = to_tensor_from_bgr(i2).to(device)
with torch.no_grad():
    logits = model(t1, t2)
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
print("Probs (different, same):", probs)
