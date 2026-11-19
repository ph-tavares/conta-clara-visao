import unittest

import numpy as np

from preparar_reais import construir_grupos, dividir_por_grupo

class GroupSplitTests(unittest.TestCase):
    def test_GroupSplit_same_digits_share_group(self):
        rotulos={'a':['1','2','3'],'b':['3','2','1'],'c':['9','9','1']}
        hashes={'a':np.zeros((4,256),bool),'b':np.ones((4,256),bool),'c':np.zeros((4,256),bool)}
        hashes['c'][:,::2]=True  # 128 bits diferente de 'a' e de 'b' em todas as rotacoes
        grupos=construir_grupos(rotulos,hashes,limite=12)
        self.assertEqual(grupos['a'],grupos['b'])
        self.assertNotEqual(grupos['a'],grupos['c'])

    def test_GroupSplit_near_identical_images_share_group(self):
        base=np.zeros((4,256),bool); base[:,::3]=True
        quase=base.copy(); quase[0,:5]=~quase[0,:5]
        outra=np.zeros((4,256),bool); outra[:,1::2]=True
        grupos=construir_grupos({'x':['1'],'y':['7'],'z':['5']},{'x':base,'y':quase,'z':outra},limite=12)
        self.assertEqual(grupos['x'],grupos['y'])
        self.assertNotEqual(grupos['x'],grupos['z'])

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
