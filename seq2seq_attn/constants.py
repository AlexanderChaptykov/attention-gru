import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.5

PRINT_EVERY = 90

MAX_LENGTH = 10
