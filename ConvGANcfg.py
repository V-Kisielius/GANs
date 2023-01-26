import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH = 16
WORKERS = 4
NOISE_DIM = 100
EPOCHS = 20
DEBUG = 100
ngf = 16
nc = 1
ndf = 16

lr = 0.0002
beta1 = 0.5