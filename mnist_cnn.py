import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the CNN architecture
class SimpleMNISTCNN(nn.Module):
    def __init__(self):
        super(SimpleMNISTCNN, self).__init__()
        # First convolutional layer (1 input channel, 10 output channels)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        # Second convolutional layer (10 input channels, 20 output channels)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # More output channels
        print(f"conv2.weight.shape: {self.conv2.weight.shape}")
        # Fully connected layer
        self.fc = nn.Linear(3200, 10)  # Match the flattened size

    def forward(self, x):#(batch_size, 1, 28, 28)
        print(f"input shape: {x.shape}")
        # First conv -> relu -> max pool
        x = self.conv1(x)#(batch_size, 16, 24, 24)
        print(f"conv1 shape: {x.shape}")
        x = F.gelu(F.max_pool2d(x, 2))#(batch_size, 16, 12, 12)
        # Second conv -> relu -> max pool
        print(f"conv2 shape: {x.shape}")
        x = self.conv2(x)
        x = F.gelu(F.max_pool2d(x, 2))
        # Flatten the tensor
        x = x.view(-1, 3200)
        # Fully connected layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Training setup
def train_mnist():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )


    # Initialize model
    model = SimpleMNISTCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Training loop
    accuracies = []  # Store accuracy for plotting
    epochs = 5
    plt.ion()
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1)  # Get predictions
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                accuracy = 100. * correct / total
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]'
                      f'\tLoss: {loss.item():.6f}'
                      f'\tAccuracy: {accuracy:.2f}%')
        
        # Epoch accuracy
        epoch_accuracy = 100. * correct / total
        accuracies.append(epoch_accuracy)
        
        # Plot accuracy
        '''
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 2), accuracies, marker='o')
        plt.title('Training Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(False)
        plt.draw()
        '''
    torch.save(model.state_dict(), 'mnist_cnn.pth')

if __name__ == '__main__':
    train_mnist()