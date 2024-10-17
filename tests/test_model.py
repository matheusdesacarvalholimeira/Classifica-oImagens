import unittest
import torch
from src.model import criar_modelo

class TestModel(unittest.TestCase):
    def test_criar_modelo(self):
        modelo = criar_modelo()
        self.assertIsNotNone(modelo)
        self.assertEqual(modelo.fc.out_features, 2)  # Ajuste para o número de classes

if __name__ == '__main__':
    unittest.main()