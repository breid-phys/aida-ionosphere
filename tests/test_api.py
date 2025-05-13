
import aida
import numpy as np
import unittest


class Test_api(unittest.TestCase):
    def test_api(self):

        Model = aida.AIDAState()

        Model.fromAPI(np.datetime64("2025-04-11T12:00:01"), 'AIDA', 'ultra')

        np.testing.assert_allclose(Model.calc(45, 55)['NmF2'], 1.30034652e+12)
