# MNIST --> Modified National Institute of Standards and Technology
# ReLU --> Rectified Linear Unit
# complete flow
"""
Input
[64, 1, 28, 28]
       ↓
Conv1
[64, 32, 26, 26]
       ↓
ReLU
       ↓
MaxPool
[64, 32, 13, 13]
       ↓
Conv2
[64, 64, 11, 11]
       ↓
ReLU
       ↓
MaxPool
[64, 64, 5, 5]
       ↓
Flatten
[64, 1600]
       ↓
Linear
[64, 10]
"""
import torch
import torch.nn as nn

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from PIL import Image,ImageOps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

train_dataset=datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])# Convert image to tensor and normalize pixel values
)

test_dataset=datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))# changed vlue from (0,255) to work in pytorch
])# Convert image to tensor and normalize pixel values
)

train_loader=DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader=DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
# print("Training samples:", len(train_dataset))
# print("Test samples:", len(test_dataset))

# image,label=train_dataset[0]
# images, labels = next(iter(train_loader))

# print("Batch image shape:", images.shape)
# print("Batch label shape:", labels.shape)

# print("Image type: ", type(image))
# print("Image shape: ", image.shape)
# print("Label: ",label)
# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(f"Label: {label}")
# plt.axis("off")
# plt.show()
# print("Minimum pixel value:", image.min())
# print("Maximum pixel value:", image.max())
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1= nn.Conv2d(# add convolution layer
            in_channels=1,
            out_channels=32,
            kernel_size=3
        )

        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)

        self.conv2=nn.Conv2d(# second convo reduce 28-13-11 and 1-32-64
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
        x=self.relu(x) #to convert -ve value to 0 and +ve remains same
        x=self.pool(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)# reduce 11 to 5

        x=self.flatten(x)# 1600--> 64*5*5 to convert feature map into vector
        x=self.fc(x)
        return x
    


model=CNN().to(device)

""" Adam is a commonly used optimizer for neural
 networks and is a reasonable choice for this beginner CNN"""

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# output=model(images)

# print("Input shape: ",images.shape)
# print("Output shape: ",output.shape)
# print("Output: ",output[0])

# # print("Minimum output value:", output.min().item())
# # print("Maximum output value:", output.max().item())

# # Step 4:- Calculate loss
# """
# CrossEntropyLoss is designed to take the raw logits directly and 
# internally performs the mathematically appropriate softmax/log 
# operations in a numerically stable way."""

criterion=nn.CrossEntropyLoss()
# loss=criterion(output,labels)
# print("Loss: ",loss.item())

# loss.backward()

# old_weight=model.fc.weight[0,0].item()

# optimizer.step()

# new_weight=model.fc.weight[0,0].item()

# print("Gradient shape:", model.fc.weight.grad.shape)
# print("Gradient sample:", model.fc.weight.grad[0][:5])

# print("Old weight:", old_weight)
# print("New weight:", new_weight)

# Step 4: Training
num_epochs=5
for epoch in range(num_epochs):
    total_loss=0

    for images,labels in train_loader:

        images=images.to(device)
        labels=labels.to(device)
        # clear gradients from previous batch
        optimizer.zero_grad()

        # Forward pass
        outputs=model(images)

        # Calculate loss
        loss=criterion(outputs,labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Add this batch's loss
        total_loss+=loss.item()

    average_loss=total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}],"
        f"Loss:{average_loss:.4f}"
    )

torch.save(model.state_dict(),"mnist_cnn.pth")
print("Model saved successfully.")

# Step 5: Evaluation
model.eval()

correct=0
total=0
all_predictions=[]
all_labels=[]

with torch.no_grad():
    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        
        outputs=model(images)

        _, predicted=torch.max(outputs,1)
        # total+= labels.size(0)
        # correct+=(predicted== labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy=100* sum(
    p==l for p,l in zip(all_predictions,all_labels)
) / len(all_labels)

print(f"Test Acuracy: {accuracy: .2f}%")

cm=confusion_matrix(all_labels,all_predictions)

print("Confusion Matrix: ")
print(cm)

plt.figure(figsize=(8,6))
plt.imshow(cm)
plt.title("MNIST Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

plt.xticks(range(10))
plt.yticks(range(10))

plt.colorbar(label="Number of Images")

# Display the count inside every cell
for i in range(10):
    for j in range(10):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

# plt.show()

# Step 6: Test Handwritten digit

image=Image.open("test_images/digit_5.jpg")

# convert to grayscale
image=image.convert("L")

#Invert colors: dark digit on light background
# becomes white digits on black background
image=ImageOps.invert(image)

# find the bounding box of the digit
bbox=image.getbbox()

# Crop around the digit
image= image.crop(bbox)

# Add some padding around the digit
padding=4
image=ImageOps.expand(
    image,
    border=padding,
    fill=0
)

#Resize to MNIST size
image=image.resize((28,28))

# convert to tensor
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

image_tensor=transform(image)

# add batch  dimension
image_tensor=image_tensor.unsqueeze(0)
plt.figure(figsize=(4, 4))

plt.imshow(image_tensor.squeeze(), cmap="gray")

plt.title("Processed Handwritten Digit")
plt.axis("off")

plt.show(block=False)
plt.pause(2)
plt.close()

print("Image tensor shape: ", image_tensor.shape)

# Make prediction

image_tensor = image_tensor.to(device)

with torch.no_grad():

    output = model(image_tensor)
    probabilities = torch.softmax(output,dim=1)
    predicted_digit = torch.argmax(output, dim=1).item()

print("Predicted digit:", predicted_digit)
print("Probabilities: ", probabilities[0])