import unittest

import numpy as np

from preparar_digitos import recorte_quadrado, digito_sintetico

class DigitCropTests(unittest.TestCase):
    def test_DigitCrop_is_square_and_sized(self):
        imagem = np.zeros((100, 200, 3), np.uint8)
        imagem[20:80, 50:80] = 255
        recorte = recorte_quadrado(imagem, (50, 20, 80, 80), lado=64)
        self.assertEqual(recorte.shape, (64, 64, 3))
        self.assertGreater(recorte[32, 32].mean(), 200)

    def test_DigitCrop_box_at_border_does_not_fail(self):
        imagem = np.full((50, 50, 3), 128, np.uint8)
        self.assertEqual(recorte_quadrado(imagem, (-5, -5, 20, 30), lado=32).shape, (32, 32, 3))

    def test_SyntheticDigit_is_deterministic(self):
        a = digito_sintetico(7, semente=3, lado=64)
        b = digito_sintetico(7, semente=3, lado=64)
        self.assertEqual(a.shape, (64, 64, 3))
        self.assertTrue(np.array_equal(a, b))

if __name__ == '__main__':
    unittest.main()
