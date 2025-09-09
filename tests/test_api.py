
import aida
import numpy as np
import unittest
import datetime
from pathlib import Path
import os

class Test_api(unittest.TestCase):
    def test_api(self):

        print(Path(os.environ['USERPROFILE']).joinpath(r"\AppData\Local\aida\\").joinpath(r"api_config.ini"))

        Model = aida.AIDAState()

        Model.fromAPI(np.datetime64("2025-04-11T12:00:01"), 'AIDA', 'ultra', APIconfig=Path(os.environ['USERPROFILE']).joinpath(r"\AppData\Local\aida\\").joinpath(r"api_config.ini"))

        np.testing.assert_allclose(Model.calc(45, 55)['NmF2'], 1.30034652e+12)

    def test_createFilenames(self):
        """


        Returns
        -------
        None.

        """

        time = None
        pattern = ''
        assert aida.createFilenames(pattern, time) == ['']

        time = datetime.datetime(2020, 1, 2, 3, 4, 5)
        pattern = '{yyyy}{yy}{mm}{doy}{dd}{HH}{H}{MM}{SS}{GPSW}{GPSD}'
        assert aida.createFilenames(pattern, time) == [
            '202020010020203d040520864']

        time = datetime.datetime(1970, 1, 1, 0, 0, 0)
        pattern = '{yyyy}{yy}{mm}{doy}{dd}{HH}{H}{MM}{SS}'
        assert aida.createFilenames(pattern, time) == ['197070010010100a0000']

        time = datetime.datetime(1980, 1, 6, 0, 0, 0)
        pattern = '{GPSW}{GPSD}'
        assert aida.createFilenames(pattern, time) == ['00000']

        time = [datetime.datetime(1980, 12, 31, 0, 0, 0),
                datetime.datetime(1980, 3, 1, 0, 0, 0),
                datetime.datetime(1981, 3, 1, 0, 0, 0)]
        pattern = '{doy}'
        assert aida.createFilenames(pattern, time) == ['366', '061', '060']

        time = np.datetime64('2003-03-04') + \
            np.array(range(24)).astype('timedelta64[h]')
        pattern = '{H}'
        assert aida.createFilenames(
            pattern,
            time) == [
            'a',
            'b',
            'c',
            'd',
            'e',
            'f',
            'g',
            'h',
            'i',
            'j',
            'k',
            'l',
            'm',
            'n',
            'o',
            'p',
            'q',
            'r',
            's',
            't',
            'u',
            'v',
            'w',
            'x']

        time = np.datetime64('1342-04-01T13:34:12.000997')
        name = 'test'
        pattern = '{SS}{MM}{name}'
        assert aida.createFilenames(pattern, time, name=name) == ['1234test']
