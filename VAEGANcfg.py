import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 64
epochs=25
lr=3e-4
alpha=0.1
gamma=15