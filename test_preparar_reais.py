import unittest

from preparar_reais import agrupar, dividir_por_grupo

class GroupSplitTests(unittest.TestCase):
    def test_GroupSplit_same_digits_share_group(self):
        grupos=agrupar({'a.jpg':['1','2','3'],'b.jpg':['3','2','1'],'c.jpg':['9','9','1']})
        self.assertEqual(grupos['a.jpg'],grupos['b.jpg'])
        self.assertNotEqual(grupos['a.jpg'],grupos['c.jpg'])

    def test_GroupSplit_no_group_crosses_splits(self):
        grupos={f'img{i}.jpg':f'g{i//3}' for i in range(300)}
        divisao=dividir_por_grupo(grupos)
        por_grupo={}
        for arquivo,parte in divisao.items():
            por_grupo.setdefault(grupos[arquivo],set()).add(parte)
        self.assertTrue(all(len(p)==1 for p in por_grupo.values()))
        self.assertEqual(set(divisao.values()),{'train','val','test'})

    def test_GroupSplit_is_deterministic(self):
        grupos={f'img{i}.jpg':f'g{i}' for i in range(50)}
        self.assertEqual(dividir_por_grupo(grupos),dividir_por_grupo(dict(reversed(list(grupos.items())))))

if __name__=='__main__':
    unittest.main()
