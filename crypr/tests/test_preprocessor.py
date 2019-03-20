"""Check That Preprocessors are Functioning Correctly"""
import unittest
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv, find_dotenv
from crypr.preprocessors import SimplePreprocessor, DWTSmoothPreprocessor


class TestPreprocessor(unittest.TestCase):
    """Base class for testing preprocessors"""
    def setUp(self):
        load_dotenv(find_dotenv())
        self.project_path = os.path.dirname(find_dotenv())
        self.data_dir = os.path.join(self.project_path, 'crypr', 'tests', 'data')
        self.data = pd.read_csv(os.path.join(self.data_dir, 'test_raw_btc.csv'), index_col=0)
        self.Tx = 72
        self.Ty = 1
        self.target_col = 'close'
        self.wavelet = 'haar'
        self.moving_averages = [6, 12, 24, 48, 72]
        self.dummy_arr_2d = np.reshape(np.arange(5*4*3), (5, 12))

    def tearDown(self):
        self.data = None
        self.Tx = None
        self.Ty = None
        self.target_col = None
        self.wavelet = None
        self.moving_averages = None

    def shapeCheck(self, X_processed):
        self.assertTrue(self.Tx == X_processed.shape[-1])
        self.assertTrue(len(X_processed.shape) == 3)

    def reshapeCheck(self, reshaped_data):
        dummy = self.dummy_arr_2d
        # Same # of values
        self.assertEqual(dummy.size, reshaped_data.size)
        # Equal sum of values
        self.assertEqual((dummy[0].sum(), dummy[0].mean(), dummy[0].std()),
                        (reshaped_data[0].sum(), reshaped_data[0].mean(), reshaped_data[0].std()))


class TestDWTSmoothPreprocessor(TestPreprocessor):
    def runTest(self):
        preprocessor = DWTSmoothPreprocessor(production=False, target_col=self.target_col, wavelet=self.wavelet,
                                             Tx=self.Tx, Ty=self.Ty, name='TestDWTSmoothPreprocessor')
        X, y = preprocessor.fit_transform(self.data)
        self.shapeCheck(X)

        preprocessor.production = True
        X = preprocessor.fit_transform(self.data)
        self.shapeCheck(X)

        dummy_reshaped = preprocessor._reshape(self.dummy_arr_2d)
        self.reshapeCheck(dummy_reshaped)


class TestSimplePreprocessor(TestPreprocessor):
    def runTest(self):
        preprocessor = SimplePreprocessor(production=False, target_col=self.target_col,
                                          moving_averages=self.moving_averages,
                                          Tx=self.Tx, Ty=self.Ty, name='TestSimplePreprocessor')
        X, y = preprocessor.fit_transform(self.data)
        self.shapeCheck(X)

        preprocessor.production = True
        X = preprocessor.fit_transform(self.data)
        self.shapeCheck(X)

        preprocessor.Tx = 3
        dummy_reshaped = preprocessor._reshape(pd.DataFrame(self.dummy_arr_2d))
        self.reshapeCheck(dummy_reshaped)
