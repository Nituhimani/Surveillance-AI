import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys

# Define the same model architecture used for training
class DNN_fixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.tied_convolution = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(20, 25, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.patch = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=5, stride=5),
            nn.ReLU(inplace=True)
        )
        self.across_patch = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(4500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 2)
        )
        self.pad = nn.ZeroPad2d(2)

    def get_f(self, f):
        return F.interpolate(f, scale_factor=5, mode='nearest')

    def get_g(self, y):
        b, c, h, w = y.shape
        y_p = self.pad(y)
        patches = F.unfold(y_p, kernel_size=5, stride=1)
        g = F.fold(patches, output_size=(h*5, w*5), kernel_size=5, stride=5)
        return g


    def cross_input_neighbourhood_difference(self, y1, y2):
        return F.relu(self.get_f(y1) - self.get_g(y2))


    def forward(self, img1, img2):
        y1 = self.tied_convolution(img1)
        y2 = self.tied_convolution(img2)
        y1_2 = self.cross_input_neighbourhood_difference(y1, y2)
        y2_1 = self.cross_input_neighbourhood_difference(y2, y1)
        y1 = self.patch(y1_2)
        y1 = self.across_patch(y1)
        y2 = self.patch(y2_1)
        y2 = self.across_patch(y2)
        y = torch.hstack((y1, y2))
        y = y.view(y.shape[0], -1)
        logits = self.fc(y)
        return logits

# Function to perform re-identification
def reidentify_images(model_path, image1_path, image2_path):
    # Define image transformations
    IMAGE_WIDTH = 60
    IMAGE_HEIGHT = 160
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor()
    ])

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNN_fixed()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess the images
    try:
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
    except FileNotFoundError as e:
        print(f"Error loading image: {e}. Please check the image paths.", file=sys.stderr)
        return None # Indicate failure

    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)

    # Get model output
    with torch.no_grad():
        outputs = model(img1_tensor, img2_tensor)

    # Interpret output
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Determine if same or different
    if predicted_class == 1:
        prediction = "SAME person"
    else:
        prediction = "DIFFERENT persons"

    return prediction, probabilities.squeeze().tolist()

if __name__ == "__main__":
    # Example usage:
    # Replace with the actual path to your checkpoint file and test images
    model_checkpoint_path = '/content/checkpoint_epoch26.pth'
    test_image1_path = '/content/5_039_2_09.png' # Replace with your image path
    test_image2_path = '/content/5_039_2_10.png' # Replace with your image path

    result, probabilities = reidentify_images(model_checkpoint_path, test_image1_path, test_image2_path)

    if result:
        print(f"The model predicts that the two images belong to: {result}")
        print(f"Probabilities (Different, Same): {probabilities}")
