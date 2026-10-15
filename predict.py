import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from torchvision import transforms


# ==========================================
# 1. DEVICE CONFIGURATION
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. CNN MODEL ARCHITECTURE
# ==========================================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

model_path = "mnist_cnn.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Trained model not found: {model_path}"
    )

model = CNN().to(device)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.eval()


# ==========================================
# 4. LOAD ORIGINAL IMAGE
# ==========================================

image_path = "test_images/digit_5.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"Input image not found: {image_path}"
    )

original_image = Image.open(image_path).convert("L")


# ==========================================
# 5. IMAGE PREPROCESSING
# ==========================================

# Invert: dark handwriting becomes white
image = ImageOps.invert(original_image)

# Threshold to remove low-intensity background pixels
image = image.point(
    lambda p: 255 if p > 128 else 0
)

# Detect the digit's bounding box
bbox = image.getbbox()

if bbox is None:
    raise ValueError(
        "No handwritten digit detected in the image."
    )

# Crop around the digit
image = image.crop(bbox)

# Add padding around the digit
image = ImageOps.expand(
    image,
    border=4,
    fill=0
)

# Resize to MNIST dimensions
image = image.resize(
    (28, 28),
    Image.Resampling.LANCZOS
)


# ==========================================
# 6. CONVERT IMAGE TO TENSOR
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])

image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0).to(device)


# ==========================================
# 7. PREDICTION
# ==========================================

with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_digit = output.argmax(dim=1).item()

    confidence = (
        probabilities[0, predicted_digit].item() * 100
    )


# ==========================================
# 8. TERMINAL OUTPUT
# ==========================================

print("\n" + "=" * 48)
print("       MNIST DIGIT RECOGNITION SYSTEM")
print("=" * 48)

print(f"  Input Image      : {os.path.basename(image_path)}")
print(f"  Predicted Digit  : {predicted_digit}")
#print(f"  Model Confidence : {confidence:.2f}%")
print("  Architecture     : Convolutional Neural Network")
print("  Framework        : PyTorch")
print(f"  Execution Device : {device}")

print("=" * 48)
print("       PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 48 + "\n")


# ==========================================
# 9. PROFESSIONAL VISUALIZATION
# ==========================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 5),
    gridspec_kw={"width_ratios": [1, 1]}
)

fig.canvas.manager.set_window_title(
    "MNIST Handwritten Digit Recognition"
)

fig.suptitle(
    "MNIST HANDWRITTEN DIGIT RECOGNITION",
    fontsize=17,
    fontweight="bold",
    y=0.96
)


# ------------------------------------------
# LEFT: ORIGINAL HANDWRITTEN IMAGE
# ------------------------------------------

axes[0].imshow(
    original_image,
    cmap="gray"
)

axes[0].set_title(
    "Original Handwritten Image",
    fontsize=12,
    fontweight="bold",
    pad=12
)

axes[0].axis("off")


# ------------------------------------------
# RIGHT: PROCESSED MNIST IMAGE
# ------------------------------------------

axes[1].imshow(
    image,
    cmap="gray",
    interpolation="nearest",
    vmin=0,
    vmax=255
)

axes[1].set_title(
    "Processed 28 × 28 Image",
    fontsize=12,
    fontweight="bold",
    pad=12
)

axes[1].axis("off")


# ------------------------------------------
# PREDICTION RESULTS
# ------------------------------------------

fig.text(
    0.5,
    0.16,
    f"Predicted Digit: {predicted_digit}",
    ha="center",
    fontsize=21,
    fontweight="bold"
)

fig.text(
    0.5,
    0.09,
    f"Model Confidence: {confidence:.2f}%",
    ha="center",
    fontsize=15
)

fig.text(
    0.5,
    0.035,
    "Powered by PyTorch CNN",
    ha="center",
    fontsize=11
)


# ------------------------------------------
# FINAL DISPLAY
# ------------------------------------------

plt.tight_layout(
    rect=[0.02, 0.21, 0.98, 0.88]
)

plt.show()