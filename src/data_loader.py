import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.config import Config

def carregar_dados():
    transformacoes = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalização para modelos pré-treinados
    ])

    # Carregar os conjuntos de treino e validação
    conjunto_treino = ImageFolder(os.path.join(Config.DATA_DIR, 'train'), transform=transformacoes)
    conjunto_val = ImageFolder(os.path.join(Config.DATA_DIR, 'val'), transform=transformacoes)

    # Criar DataLoaders
    loader_treino = DataLoader(conjunto_treino, batch_size=Config.BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(conjunto_val, batch_size=Config.BATCH_SIZE, shuffle=False)

    return loader_treino, loader_val