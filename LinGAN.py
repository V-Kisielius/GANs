import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import clear_output
from LinGANcfg import device, NOISE_DIM

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential( 
            nn.Linear(NOISE_DIM, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 28*28),
            nn.Sigmoid())
        
    def forward(self, x):
        return self.model(x)

    def plot_results(self):
        # plot 10x10 images
        plt.figure(figsize=(15,15))
        for i in range(100):
            noise = torch.randn(1, NOISE_DIM).to(device)
            fake = self(noise)
            plt.subplot(10,10,i+1)
            plt.imshow(fake.cpu().detach().numpy().squeeze().reshape(28, 28), cmap='gray')
            plt.axis('off')
        plt.show()

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1), 
            nn.Sigmoid())
    def forward(self, x):
        return self.model(x)

def train_GAN(generator, discriminator, criterion, optim_G, optim_D, dataloader, epochs=10, debug=100):
    history_G = []
    history_D = []
    for epoch in range(epochs):
        for i, (img, _) in enumerate(dataloader):
            # train discriminator
            optim_D.zero_grad()
            # real
            real = img.view(img.size(0), -1).to(device)
            real_label = torch.ones(real.size(0), 1).to(device)
            real_out = discriminator(real)
            loss_real = criterion(real_out, real_label)
            # fake
            noise = torch.randn(real.size(0), NOISE_DIM).to(device)
            fake = generator(noise)
            fake_label = torch.zeros(real.size(0), 1).to(device)
            fake_out = discriminator(fake.detach())
            loss_fake = criterion(fake_out, fake_label)
            # backward
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optim_D.step()
            
            # train generator
            optim_G.zero_grad()
            fake_label = torch.ones(real.size(0), 1).to(device)
            fake_out = discriminator(fake)
            loss_G = criterion(fake_out, fake_label)
            loss_G.backward()
            optim_G.step()
            
            history_G.append(loss_G.item())
            history_D.append(loss_D.item())

            if i % debug == 0:
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
                    plt.imshow(fake[j].cpu().detach().numpy().squeeze().reshape(28, 28), cmap='gray')
                    plt.axis('off')
                plt.suptitle(f'Epoch: {epoch+1}/{epochs}\nBatch: {i}/{len(dataloader)}\nLoss_G: {loss_G.item():.4f}\nLoss_D: {loss_D.item():.4f}')
                plt.show()