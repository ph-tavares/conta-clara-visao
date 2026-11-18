import unittest

from conta_clara import Digit, read_digits

class ReadingTests(unittest.TestCase):
    def sample(self, confidence=0.9):
        return [Digit(text, confidence, (14*i, 10, 14*i+9, 30)) for i,text in enumerate('00731')]

    def test_order_and_leading_zeros(self):
        reading=read_digits(list(reversed(self.sample())))
        self.assertEqual(reading['leitura_candidata_kwh'],'00731')
        self.assertEqual(reading['status'],'leitura_sugerida')

    def test_low_score_requires_review(self):
        result=read_digits(self.sample(.60))
        self.assertEqual(result['status'],'confirmacao_necessaria')
        self.assertEqual(result['motivo'],'baixa_confianca')

    def test_incomplete_has_no_candidate(self):
        result=read_digits(self.sample()[:-1])
        self.assertIsNone(result['leitura_candidata_kwh'])
        self.assertEqual(result['motivo'],'quantidade_de_digitos')

    def test_inconsistent_alignment_requires_review(self):
        digits=self.sample()
        digits[4]=Digit('1',.93,(56,100,65,120))
        self.assertEqual(read_digits(digits)['motivo'],'disposicao_inconsistente')

if __name__=='__main__':
    unittest.main()
