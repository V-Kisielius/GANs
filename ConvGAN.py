import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import clear_output
from ConvGANcfg import device, NOISE_DIM, ngf, nc, ndf

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( NOISE_DIM, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            #nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            #nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 8, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        
    def forward(self, x):
        return self.model(x)

    def plot_results(self):
        # plot 10x10 images
        plt.figure(figsize=(15,15))
        for i in range(100):
            noise = torch.randn(1, NOISE_DIM, 1, 1).to(device)
            fake = self(noise)
            plt.subplot(10,10,i+1)
            plt.imshow(fake.cpu().detach().numpy().squeeze().reshape(32, 32), cmap='gray')
            plt.axis('off')
        plt.show()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            #nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        #print(x.shape)
        return self.model(x)

def train_GAN(generator, discriminator, criterion, optim_G, optim_D, dataloader, epochs=10, debug=100):
    history_D = []
    history_G = []
    for epoch in range(epochs):
        # use device
        for i, (img, _) in enumerate(dataloader):
            img = img.to(device)
            # train discriminator
            optim_D.zero_grad()
            # real
            real = discriminator(img)
            real_loss = criterion(real, torch.ones_like(real))
            # fake
            noise = torch.randn((img.shape[0], NOISE_DIM, 1, 1)).to(device)
            fake = generator(noise)
            Df = discriminator(fake)
            fake_loss = criterion(Df, torch.zeros_like(Df))
            # update discriminator
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optim_D.step()
            # train generator
            optim_G.zero_grad()
            fake = generator(noise)
            Df = discriminator(fake)
            g_loss = criterion(Df, torch.ones_like(Df))
            g_loss.backward()
            optim_G.step()
            # update history
            history_D.append(d_loss.item())
            history_G.append(g_loss.item())
            
            # plot losses and generated images using gridspec
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
                    plt.imshow(fake[j].cpu().detach().numpy().squeeze().reshape(32, 32), cmap='gray')
                    plt.axis('off')
                plt.suptitle(f'Epoch: {epoch+1}/{epochs}\nBatch: {i}/{len(dataloader)}\nLoss_G: {g_loss.item():.4f}\nLoss_D: {d_loss.item():.4f}')
                plt.show()
