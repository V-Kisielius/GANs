import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH = 16
WORKERS = 4
NOISE_DIM = 50
EPOCHS = 20
DEBUG = 100

lr_D = 1e-4
lr_G = 1e-4