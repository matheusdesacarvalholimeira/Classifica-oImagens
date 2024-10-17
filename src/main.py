import torch
from src.data_loader import carregar_dados
from src.model import criar_modelo, treinar_modelo
from src.config import Config

def main():
    loader_treino, loader_val = carregar_dados()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = criar_modelo()
    modelo_treinado = treinar_modelo(modelo, loader_treino, loader_val, device)
    torch.save(modelo_treinado.state_dict(), f"{Config.MODEL_DIR}/modelo_treinado.pth")
    print("Modelo treinado e salvo.")

if __name__ == "__main__":
    main()