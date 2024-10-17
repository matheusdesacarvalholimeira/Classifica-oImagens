import os
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

class Config:
    DATA_DIR = os.getenv("DATA_DIR")
    MODEL_DIR = os.getenv("MODEL_DIR")
    NUM_CLASSES = 2                 
    IMAGE_SIZE = 224                 
    BATCH_SIZE = 32                 
    EPOCHS = 10                     
    LEARNING_RATE = 0.001           