in order to run script integrated_detector_reid.py you will need to:

# 1. Install PyTorch (choose CUDA or CPU)
# If you have NVIDIA GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# If only CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install Ultralytics YOLOv8 (includes YOLO, tracking, utils)
pip install ultralytics

# 3. Install OpenCV for video processing
pip install opencv-python

# 4. Install NumPy
pip install numpy

# 5. Install LAP (needed for ByteTrack)
pip install lap>=0.5.12
