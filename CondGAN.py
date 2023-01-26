import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import clear_output

class DiscriminatorForMNIST(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network discriminator.
    Args:
        image_size (int): The size of the image. (Default: 28)
        channels (int): The channels of the image. (Default: 1)
        num_classes (int): Number of classes for dataset. (Default: 10)
    """
    def __init__(self, image_size: int = 28, channels: int = 1, num_classes: int = 10) -> None:
        super(DiscriminatorForMNIST, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(channels * image_size * image_size + num_classes, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, inputs: torch.Tensor, labels: list = None) -> torch.Tensor:
        r""" Defines the computation performed at every call.
        Args:
            inputs (tensor): input tensor into the calculation.
            labels (list):  input tensor label.
        Returns:
            A four-dimensional vector (N*C*H*W).
        """
        inputs = torch.flatten(inputs, 1)
        conditional = self.label_embedding(labels)
        conditional_inputs = torch.cat([inputs, conditional], dim=-1)
        out = self.main(conditional_inputs)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class Generator(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator.
    Args:
        image_size (int): The size of the image. (Default: 28)
        channels (int): The channels of the image. (Default: 1)
        num_classes (int): Number of classes for dataset. (Default: 10)
    """

    def __init__(self, image_size: int = 28, channels: int = 1, num_classes: int = 10) -> None:
        super(Generator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(100 + num_classes, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(1024, channels * image_size * image_size),
            nn.Tanh()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, inputs: torch.Tensor, labels: list = None) -> torch.Tensor:
        """
        Args:
            inputs (tensor): input tensor into the calculation.
            labels (list):  input tensor label.
        Returns:
            A four-dimensional vector (N*C*H*W).
        """

        conditional_inputs = torch.cat([inputs, self.label_embedding(labels)], dim=-1)
        out = self.main(conditional_inputs)
        out = out.reshape(out.size(0), self.channels, self.image_size, self.image_size)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def train_GAN(D, G, criterion, num_epochs, batch_size, device, dataloader, optimizer_D, optimizer_G):
    history_D = []
    history_G = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # Create the labels which are later used as input for the BCE loss
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise as generator input
            z = torch.randn(batch_size, 100).to(device)

            # Generate a batch of images
            fake_images = G(z, labels)

            # Real images
            real_loss = criterion(D(images, labels), real_labels)
            # Fake images
            fake_loss = criterion(D(fake_images.detach(), labels), fake_labels)
            # Total loss
            d_loss = (real_loss + fake_loss) / 2

            # Backprop
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            gen_images = G(z, labels)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            g_loss = criterion(D(gen_images, labels), real_labels)

            # Backprop
            g_loss.backward()
            optimizer_G.step()

            history_D.append(d_loss.item())
            history_G.append(g_loss.item())

            if i % 100 == 0:
                clear_output(True)
                # plot losses and generated images using gridspec
                gs = gridspec.GridSpec(2, 4)
                plt.figure(figsize=(15,10))
                plt.subplot(gs[:, :2])
                plt.plot(history_G, label='Generator')
                plt.plot(history_D, label='Discriminator')
                plt.legend()
                for j in range(4):
                    plt.subplot(gs[j // 2, 2 + j % 2])
                    plt.imshow(gen_images[j].cpu().detach().numpy().squeeze().reshape(28, 28), cmap='gray')
                    plt.axis('off')
                plt.suptitle(f'Epoch: {epoch+1}/{num_epochs}\nBatch: {i}/{len(dataloader)}\nLoss_G: {g_loss.item():.4f}\nLoss_D: {d_loss.item():.4f}')
                plt.show()




                
