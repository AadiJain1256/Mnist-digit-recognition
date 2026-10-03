import torch
import torch.nn as nn
from PIL import Image,ImageOps
import torchvision.transforms as transforms

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1=nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3
        )

        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)

        self.conv2=nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )

        self.flatten=nn.Flatten()

        self.fc=nn.Linear(
            in_features=1600,
            out_features=10
        )

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.flatten(x)
        x=self.fc(x)

        return x

# Create model
model=CNN().to(device)

model.load_state_dict(
    torch.load("mnist_cnn.pth",map_location=device)
)

# Evaluation mode
model.eval()

# print("Model loaded successfully!")
# print("Device: ",device)

# Load
image=Image.open("test_images/digit_2.jpeg")

# grayscale
image= image.convert("L")

# invert
image=ImageOps.invert(image)
print("Image inverted successfully")

# threshold
image=image.point(lambda p : 255 if p > 128 else 0)
print("After grayscale:",image.mode)
print("image size:",image.size)


print("Original image mode:",image.mode)
print("Original image size:",image.size)

bbox=image.getbbox()

print("Bounding box:",bbox)

#Crop to the digit
image=image.crop(bbox)

image=ImageOps.expand(
    image,
    border=4,
    fill=0
)

print("After padding:", image.size)

image=image.resize((28,28))
print("After resize:",image.size)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081))])
image_tensor=transform(image)

image_tensor=image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)

# Run the model
with torch.no_grad():
    output= model(image_tensor)
print("Output shape:",output.shape)
print("Output:",output)

print("Input device:", image_tensor.device)

print("Input shape:",image_tensor.shape)


predicted_digit=output.argmax(dim=1).item()

print("Predicted digit:", predicted_digit)
