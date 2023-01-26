import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH = 16
WORKERS = 4
EPOCHS = 20
