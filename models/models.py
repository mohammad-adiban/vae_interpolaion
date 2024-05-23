import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 64)  # 20-dimensional latent space

        # Decoder layers
        self.fc3 = nn.Linear(64, 256)
        self.relu4 = nn.ReLU()
        self.fc4 = nn.Linear(256, 32 * 7 * 7)
        self.relu5 = nn.ReLU()
        self.conv_t1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.relu6 = nn.ReLU()
        self.conv_t2 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        mu, logvar = self.fc2(x), self.fc2(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        z = self.fc3(z)
        z = self.relu4(z)
        z = self.fc4(z)
        z = self.relu5(z)
        z = z.view(-1, 32, 7, 7)
        z = self.conv_t1(z)
        z = self.relu6(z)
        z = self.conv_t2(z)
        x = self.sigmoid(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    
# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
    

# Define the Deep CNN architecture
class CNN_deep(nn.Module):
    def __init__(self):
        super(CNN_deep, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # Additional convolutional layer
        self.relu3 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Adjusted input size for the fully connected layer
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)  # Added convolutional layer
        x = self.relu3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x




class V_CNN(nn.Module):
    def __init__(self):
        super(V_CNN, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)  # Update the input size
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Define the CNN + Dropout architecture with dropout
class CNN_d(nn.Module):
    def __init__(self):
        super(CNN_d, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)  # Added dropout
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)  # Added dropout
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)  # Added dropout
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)  # Added dropout
        x = self.fc2(x)
        return x