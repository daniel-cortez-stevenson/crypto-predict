''' Check That Preprocessors are Functioning Correctly '''

import unittest
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from crypr.base.preprocessors import *

class TestPreprocessor(unittest.TestCase):
    ''' Base class for testing preprocessors '''

    def setUp(self):
        ''' Define some unique data for validation '''
        load_dotenv(find_dotenv())
        project_path = os.path.dirname(find_dotenv())
        self.data = pd.read_csv('{}/crypr/tests/test_raw_btc.csv'.format(project_path), index_col=0)
        self.Tx = 72
        self.Ty = 1
        self.target_col = 'close'
        self.wavelet = 'haar'

    def tearDown(self):
        ''' Destroy unique data '''
        self.data = None
        self.Tx = None
        self.Ty = None

    def shapeCheck(self, X_processed):
        data_shape=X_processed.shape
        self.assertTrue(self.Tx == data_shape[-1],
                        print("Shape {} does not have Tx={} in it it's last dim".format(data_shape, self.Tx)))


class TestDWTSmoothPreprocessor(TestPreprocessor):

    def runTest(self):
        preprocessor = DWTSmoothPreprocessor(production=False, target_col=self.target_col, wavelet=self.wavelet,
                                             Tx=self.Tx, Ty=self.Ty, name='TestDWTSmoothPreprocessor')
        X, y = preprocessor.fit(self.data).transform(self.data)
        self.shapeCheck(X)

        preprocessor.production = True
        X = preprocessor.fit(self.data).transform(self.data)
        self.shapeCheck(X)

class TestSimplePreprocessor(TestPreprocessor):

    def runTest(self):
        preprocessor = SimplePreprocessor(production=False, target_col=self.target_col, moving_averages=[6,12,24,48,72],
                                             Tx=self.Tx, Ty=self.Ty, name='TestSimplePreprocessor')
        X, y = preprocessor.fit(self.data).transform(self.data)
        self.shapeCheck(X)

        preprocessor.production = True
        X = preprocessor.fit(self.data).transform(self.data)
        self.shapeCheck(X)
