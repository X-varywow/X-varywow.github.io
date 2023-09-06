Sure, here's an example of how to implement an autoencoder using PyTorch:

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoding
        x = self.encoder(x)
        
        # Decoding
        x = self.decoder(x)
        
        return x
```

In this example, we define an autoencoder with an encoder that takes an input of size 784 (the number of pixels in an image) and outputs a latent vector of size 128. The decoder then takes this latent vector and reconstructs the original input image. We use ReLU activation functions in the encoder and decoder, and a sigmoid activation function in the final layer of the decoder to ensure that the output is between 0 and 1 (since images are represented as probabilities).

To train the autoencoder, we can use a loss function such as mean squared error (MSE) to compare the reconstructed image to the original input image. We can then use an optimizer such as Adam to update the weights of the autoencoder based on the loss. Here's an example of how to train the autoencoder on MNIST:

```python
import torch.optim as optim

# Load MNIST dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=128, shuffle=True)

# Initialize autoencoder
autoencoder = Autoencoder()

# Define optimizer and loss function
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train autoencoder
for epoch in range(10):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        
        # Forward pass
        output = autoencoder(img)
        
        # Compute loss
        loss = criterion(output, img)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update weights
        optimizer.step()
        
    print('Epoch:', epoch+1, 'Loss:', loss.item())
```

In this example, we use the Adam optimizer with a learning rate of 0.001 to update the weights of the autoencoder. We also use the mean squared error loss function to compare the reconstructed image to the original input image. We train the autoencoder for 10 epochs on the MNIST dataset, and print the loss after each epoch. 