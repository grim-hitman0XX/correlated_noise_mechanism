import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from opacus.validators import ModuleValidator
from privacy_engine import CNMEngine
from tqdm.auto import tqdm
from model_zoo import MLP, CNN
from train_utils import accuracy, train, test, train_no_dp

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(
    root="./", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./", train=False, download=True, transform=transform
)

total_size = X_train.shape[0]
grad_norm = 1
epsilon = 8
delta = np.power(X_train.shape[0], -1.1)
batch_size = 1024

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# model = MLP(input_size = 784, hidden_size = 100, output_size = 10)
model = CNN()
ModuleValidator.validate(model, strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1)
EPOCHS = 50

privacy_engine = CNMEngine()
#print(CNMEngine.mro())
EPOCHS = 20
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=EPOCHS,
    target_epsilon=epsilon,
    target_delta=delta,
    max_grad_norm=grad_norm,
    mode="BLT",
    participation="minSep",
    error_type="rmse",
    d = 4,
    b = 5,
    k = 8,
)

for epoch in tqdm(range(int(EPOCHS)), desc="Epoch", unit="epoch"):
    _ = train_no_dp(model, train_loader, optimizer, criterion, epoch + 1, device)

top1_acc = test(model, test_loader, criterion, device)
