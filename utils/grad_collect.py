import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the CIFAR-10 dataset
transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# Define the ResNet18 model
model = models.resnet18(pretrained=False).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Function to save gradients
def save_gradients(epoch, iteration, gradients):
    df = pd.DataFrame(gradients)
    df.to_csv(f"gradients_epoch_{epoch}_iter_{iteration}.csv", index=False)

# Training loop
num_epochs = 2  # Example for 2 epochs, adjust as needed
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Collect gradients
        gradients = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients.append({
                    'name': name,
                    'grad': param.grad.cpu().numpy().flatten()
                })

        # Save gradients to CSV
        save_gradients(epoch, i, gradients)

        # Print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # Print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print('Finished Training')
