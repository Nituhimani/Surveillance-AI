"""
reidentify_images_full.py

Single-file tool to load the DNN_fixed re-identification model, robustly load checkpoints,
run inference on two images, and optionally create dummy checkpoints/images for quick tests.

Usage examples:
    # Run inference with a checkpoint and two images
    python reidentify_images_full.py --model model.pth --img1 face_a.png --img2 face_b.png

    # Create dummy checkpoint and images, then run the test
    python reidentify_images_full.py --create_dummy_ckpt dummy.pth --create_dummy_images img1.png img2.png --model dummy.pth --img1 img1.png --img2 img2.png

Options:
    --model PATH            : path to checkpoint (state_dict or full-model). Required for inference (unless creating dummies and supplying them).
    --img1 PATH --img2 PATH : input image paths (PIL-supported). Required for inference.
    --device DEVICE         : 'cpu' or 'cuda' (default: auto detect)
    --normalize {none,imagenet}: whether to apply ImageNet normalization to inputs (default: none)
    --debug                 : print intermediate shapes & extra info
    --create_dummy_ckpt PATH: write a random-weight state_dict to PATH (for quick testing)
    --create_dummy_images A B: create two dummy RGB images (width,height = 60,160) saved as A and B

This script tries to be robust when loading checkpoints saved in different formats:
 - plain state_dict (recommended)
 - dict containing 'state_dict' or 'model_state_dict'
 - whole model saved by torch.save(model, path)

If your model was trained with input normalisation, pass --normalize imagenet (or modify the code
with training mean/std used).

"""

import argparse
import os
import sys
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ----------------------- Model (original architecture preserved) -----------------------
class DNN_fixed(nn.Module):
    def __init__(self):
        super(DNN_fixed, self).__init__()
        # tied conv block (applied to both inputs)
        self.tied_convolution = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(20, 25, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # patch conv (operates on high-res diff map)
        self.patch = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=5, stride=5),
            nn.ReLU(inplace=True)
        )

        # across-patch conv
        self.across_patch = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # final MLP head (note: flattened dim = 4500 for input size 3x160x60)
        self.fc = nn.Sequential(
            nn.Linear(4500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 2)
        )

        # used by get_g
        self.pad = nn.ZeroPad2d(2)

    def get_f(self, y: torch.Tensor) -> torch.Tensor:
        """Upsample by factor 5 (nearest). Input shape: (B, C, h, w).
        Returns shape: (B, C, h*5, w*5)
        """
        return F.interpolate(y, scale_factor=5, mode="nearest")

    def get_g(self, y: torch.Tensor) -> torch.Tensor:
        """Extracts local patches (with padding), then folds them into a higher-res grid.
        Input y has shape (B, C, h, w) coming from tied_convolution.
        Behavior reproduces the original script: pad -> unfold (5x5, stride=1) -> fold (kernel=5, stride=5)
        Output shape: (B, C, h*5, w*5)
        """
        # y: (B, C, h, w)
        b, c, h, w = y.shape
        # pad by 2 on all sides (ZeroPad2d(2)) -> padded size: (h+4, w+4)
        y_p = self.pad(y)
        # extract 5x5 patches with stride 1 -> shape (B, C*25, n_patches)
        patches = F.unfold(y_p, kernel_size=5, stride=1)
        # fold patches into output grid with kernel_size=5 and stride=5 -> output size (h*5, w*5)
        out = F.fold(patches, output_size=(h * 5, w * 5), kernel_size=5, stride=5)
        return out

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # apply tied conv to both
        y1 = self.tied_convolution(x1)
        y2 = self.tied_convolution(x2)

        # neighbourhood-difference (high-res)
        d1 = F.relu(self.get_f(y1) - self.get_g(y2))
        d2 = F.relu(self.get_f(y2) - self.get_g(y1))

        # patch conv
        p1 = self.patch(d1)
        p2 = self.patch(d2)

        # across-patch conv
        a1 = self.across_patch(p1)
        a2 = self.across_patch(p2)

        # concatenate features along channel axis and classify
        y = torch.cat((a1, a2), dim=1)
        y = y.flatten(1)
        out = self.fc(y)
        return out

# ----------------------- Helpers for checkpoint handling & inference -----------------------

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 60

def device_select(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        if prefer.lower() == 'cpu':
            return torch.device('cpu')
        if prefer.lower() == 'cuda':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _strip_module_prefix(state_dict: dict) -> dict:
    # remove 'module.' prefix if present (common when saving from DataParallel)
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state[new_key] = v
    return new_state


def _find_state_dict_in_checkpoint(ckpt) -> Optional[dict]:
    """Try to locate a state_dict inside a loaded checkpoint object/dict."""
    if isinstance(ckpt, dict):
        # common keys
        for key in ('state_dict', 'model_state_dict', 'model', 'model_state'):
            if key in ckpt:
                v = ckpt[key]
                if isinstance(v, dict):
                    return v
        # maybe the dict itself is already a state_dict
        # detect by checking for tensor-like values
        if all(isinstance(val, torch.Tensor) for val in ckpt.values()):
            return ckpt
    return None


def load_checkpoint_into_model(model: nn.Module, path: str, map_location='cpu', debug: bool = False) -> nn.Module:
    """Load a checkpoint into the model. Handles several common checkpoint formats.
    If the checkpoint contains a whole model object, returns that model instead.
    """
    if debug:
        print(f"[debug] Loading checkpoint from {path} with map_location={map_location}")

    loaded = torch.load(path, map_location=map_location)

    # If a full model object was saved (torch.save(model,...)) then loaded may be an nn.Module
    if isinstance(loaded, nn.Module):
        if debug:
            print("[debug] Checkpoint is a full nn.Module instance; using it directly.")
        return loaded

    # Try to find a state dict inside
    state = _find_state_dict_in_checkpoint(loaded)

    if state is None:
        # As a last resort, if loaded is a dict but not state-dict-like, try heuristics
        if isinstance(loaded, dict):
            # sometimes checkpoint contains nested keys like {'net': {..}}
            for v in loaded.values():
                if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                    state = v
                    break

    if state is None:
        raise RuntimeError("Could not find a state_dict in the checkpoint. Please provide a state_dict or whole-model checkpoint.")

    # strip possible 'module.' prefixes
    state = _strip_module_prefix(state)

    # try loading
    try:
        model.load_state_dict(state)
        if debug:
            print("[debug] Loaded state_dict into model successfully.")
        return model
    except Exception as e:
        # give a helpful error message
        raise RuntimeError(f"Error loading state_dict into model: {e}")


def reidentify_images(model_path: str, img1_path: str, img2_path: str, device: Optional[str] = None,
                       normalize: str = 'none', debug: bool = False) -> Tuple[str, list]:
    """Load model from model_path (state_dict or full-model) and run inference on two images.
    Returns (result_string, [p0,p1]) where p0,p1 are probabilities for class 0 and 1.
    """
    dev = device_select(device)

    # build transforms
    tr = [transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), transforms.ToTensor()]
    if normalize == 'imagenet':
        tr.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(tr)

    # create model & load
    model = DNN_fixed()
    model = model.to(dev)

    model = load_checkpoint_into_model(model, model_path, map_location=dev, debug=debug)
    model.to(dev)
    model.eval()

    # load images
    if not os.path.exists(img1_path):
        raise FileNotFoundError(f"Image 1 not found: {img1_path}")
    if not os.path.exists(img2_path):
        raise FileNotFoundError(f"Image 2 not found: {img2_path}")

    im1 = Image.open(img1_path).convert('RGB')
    im2 = Image.open(img2_path).convert('RGB')

    t1 = transform(im1).unsqueeze(0).to(dev)
    t2 = transform(im2).unsqueeze(0).to(dev)

    if debug:
        print(f"[debug] t1 shape: {t1.shape}, t2 shape: {t2.shape}, device: {dev}")

    with torch.no_grad():
        logits = model(t1, t2)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0].tolist()
        predicted = int(torch.argmax(torch.tensor(probs)).item())

    result = 'SAME person' if predicted == 1 else 'DIFFERENT persons'
    if debug:
        print(f"[debug] logits: {logits}, probs: {probs}, predicted: {predicted}")

    return result, probs

# ----------------------- Utilities to create dummy checkpoint/images for quick tests -----------------------

def create_dummy_checkpoint(path: str):
    """Create and save a random-weight state_dict for quick testing.
    This is NOT a trained model — only useful to verify the code pipeline runs.
    """
    model = DNN_fixed()
    torch.save(model.state_dict(), path)
    print(f"Saved dummy random state_dict to {path}")


def create_dummy_images(path1: str, path2: str, color1=(128,128,128), color2=(10,120,200)):
    # PIL expects size as (width, height)
    size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    im1 = Image.new('RGB', size, color=color1)
    im2 = Image.new('RGB', size, color=color2)
    im1.save(path1)
    im2.save(path2)
    print(f"Saved dummy images: {path1}, {path2} (size={size})")


# ----------------------- CLI -----------------------
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Re-identification inference tool (single-file)')
#     parser.add_argument('--model', type=str, help='Path to model checkpoint (state_dict or full model)')
#     parser.add_argument('--img1', type=str, help='Path to image 1')
#     parser.add_argument('--img2', type=str, help='Path to image 2')
#     parser.add_argument('--device', type=str, default=None, help="'cpu' or 'cuda' (auto detect by default)")
#     parser.add_argument('--normalize', choices=['none', 'imagenet'], default='none', help='Apply normalization to inputs')
#     parser.add_argument('--debug', action='store_true', help='Print debug info')
#     parser.add_argument('--create_dummy_ckpt', type=str, help='Create a dummy random state_dict at this path')
#     parser.add_argument('--create_dummy_images', nargs=2, metavar=('OUT1','OUT2'), help='Create two dummy images (saves to these paths)')
#
#     args = parser.parse_args()
#
#     # create dummy ckpt or images if requested
#     if args.create_dummy_ckpt:
#         create_dummy_checkpoint(args.create_dummy_ckpt)
#
#     if args.create_dummy_images:
#         out1, out2 = args.create_dummy_images
#         create_dummy_images(out1, out2)
#
#     # if model and images are supplied, run inference
#     if args.model and args.img1 and args.img2:
#         try:
#             res, probs = reidentify_images(args.model, args.img1, args.img2, device=args.device,
#                                           normalize=args.normalize, debug=args.debug)
#             print("\n=== RESULT ===")
#             print(f"Prediction: {res}")
#             print(f"Probabilities: class0={probs[0]:.4f}, class1={probs[1]:.4f}")
#         except Exception as e:
#             print(f"Error during inference: {e}", file=sys.stderr)
#             if args.debug:
#                 raise
#     else:
#         print("No inference performed. To run inference provide --model MODEL --img1 IMG1 --img2 IMG2")
#         print("You can also create dummy assets with --create_dummy_ckpt and --create_dummy_images")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Re-identification inference tool')
    parser.add_argument('--model', type=str, default=r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\models\checkpoint_epoch25.pth", help='Path to model checkpoint')
    parser.add_argument('--img1', type=str, default=r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\dataset\images_detected\1_009_1_01.png", help='Path to image 1')
    parser.add_argument('--img2', type=str, default=r"C:\Users\rohan_075b4dd\OneDrive\Desktop\sih\dataset\images_detected\1_009_2_10.png", help='Path to image 2')
    parser.add_argument('--device', type=str, default=None, help="'cpu' or 'cuda'")
    parser.add_argument('--normalize', choices=['none', 'imagenet'], default='none', help='Input normalization')
    parser.add_argument('--create_dummy_ckpt', type=str, help='Create dummy checkpoint at this path')
    parser.add_argument('--create_dummy_images', nargs=2, metavar=('OUT1','OUT2'), help='Create two dummy images')

    args = parser.parse_args()

    if args.create_dummy_ckpt:
        create_dummy_checkpoint(args.create_dummy_ckpt)
    if args.create_dummy_images:
        create_dummy_images(*args.create_dummy_images)

    if args.model and args.img1 and args.img2:
        res, probs = reidentify_images(args.model, args.img1, args.img2,
                                       device=args.device, normalize=args.normalize)
        print("\n=== RESULT ===")
        print(f"Prediction: {res}")
        print(f"Probabilities: class0={probs[0]:.4f}, class1={probs[1]:.4f}")
