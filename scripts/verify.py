import torch

ckpt = torch.load(r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\models\checkpoint_epoch25.pth", map_location="cpu")

print(type(ckpt))
if isinstance(ckpt, dict):
    print(ckpt.keys())
else:
    print(ckpt)
