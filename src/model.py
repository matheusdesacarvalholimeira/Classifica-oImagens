import torch
import torch.nn as nn
import torchvision.models as models
from src.config import Config

def criar_modelo():
    modelo = models.resnet18(pretrained=True)  # Usando ResNet18 como exemplo
    for param in modelo.parameters():
        param.requires_grad = False
    modelo.fc = nn.Linear(modelo.fc.in_features, Config.NUM_CLASSES)

    return modelo
def treinar_modelo(modelo, loader_treino, loader_val, device):
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=Config.LEARNING_RATE)

    modelo.to(device)

    for epoca in range(Config.EPOCHS):
        modelo.train()
        for imagens, rotulos in loader_treino:
            imagens, rotulos = imagens.to(device), rotulos.to(device)
            otimizador.zero_grad()
            saidas = modelo(imagens)
            perda = criterio(saidas, rotulos)
            perda.backward()
            otimizador.step()

        print(f'Época [{epoca + 1}/{Config.EPOCHS}], Perda: {perda.item():.4f}')

    return modelo
