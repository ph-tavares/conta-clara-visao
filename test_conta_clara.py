import unittest

from conta_clara import Digit, read_digits, combinar

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

    def test_VariableLength_reads_seven_digits(self):
        digits=[Digit(t,.9,(14*i,10,14*i+9,30)) for i,t in enumerate('0043358')]
        result=read_digits(list(reversed(digits)))
        self.assertEqual(result['leitura_candidata_kwh'],'0043358')
        self.assertEqual(result['status'],'leitura_sugerida')

    def test_VariableLength_expected_count_is_enforced(self):
        digits=[Digit(t,.9,(14*i,10,14*i+9,30)) for i,t in enumerate('004335')]
        self.assertEqual(read_digits(digits,esperados=5)['motivo'],'quantidade_de_digitos')
        self.assertEqual(read_digits(digits,esperados=6)['leitura_candidata_kwh'],'004335')

    def test_VariableLength_more_than_eight_requires_review(self):
        digits=[Digit(t,.9,(14*i,10,14*i+9,30)) for i,t in enumerate('123456789')]
        self.assertEqual(read_digits(digits)['motivo'],'quantidade_de_digitos')

    def tilted(self,text,dy=0):
        out=[]
        for i,t in enumerate(text):
            cx,cy=20+16*i,20+12*i
            out.append(Digit(t,.9,(cx-5,cy-10+(dy if i==2 else 0),cx+5,cy+10+(dy if i==2 else 0))))
        return out

    def test_TiltedMeter_orders_along_text_axis(self):
        result=read_digits(list(reversed(self.tilted('071245'))))
        self.assertEqual(result['leitura_candidata_kwh'],'071245')
        self.assertEqual(result['status'],'leitura_sugerida')

    def test_TiltedMeter_digit_off_line_requires_review(self):
        self.assertEqual(read_digits(self.tilted('071245',dy=40))['motivo'],'disposicao_inconsistente')

    def test_TwoStage_combines_detector_and_classifier(self):
        digits=combinar([((0,0,10,20),.9),((12,0,22,20),.8)],[('4',.5),('7',1.0)])
        self.assertEqual([d.text for d in digits],['4','7'])
        self.assertAlmostEqual(digits[0].confidence,.45)
        self.assertAlmostEqual(digits[1].confidence,.8)
        self.assertEqual(digits[1].bounds,(12,0,22,20))

if __name__=='__main__':
    unittest.main()
