from unittest import TestCase
from numpy.testing import assert_array_equal
from os.path import join
import numpy as np
import pandas as pd
from crypr.util import get_project_path
from crypr.build import series_to_supervised


class TestSeriesTo(TestCase):
    def setUp(self):
        self.data_dir = join(get_project_path(), 'crypr', 'tests', 'data')
        self.test_series = pd.Series(list(range(10)))
        self.tx = 4
        self.ty_single = 1
        self.ty_multiple = 2
        self.to_supervised_ty_single = series_to_supervised(self.test_series, n_in=self.tx,
                                                            n_out=self.ty_single, dropnan=True)
        self.to_supervised_ty_multiple = series_to_supervised(self.test_series, n_in=self.tx,
                                                              n_out=self.ty_multiple, dropnan=True)

    def test_ty_single_shape(self):
        expected_shape = (self.test_series.size - self.tx, self.tx + 1)
        self.assertEqual(self.to_supervised_ty_single.shape, expected_shape)

    def test_ty_multiple_shape(self):
        expected_shape = (self.test_series.size - self.tx - (self.ty_multiple - 1), self.tx + self.ty_multiple)
        self.assertEqual(self.to_supervised_ty_multiple.shape, expected_shape)

    def test_ty_single_values(self):
        expected_values = np.array([list(range(n - self.tx, n + 1))
                                    for n in
                                    range(self.tx, self.test_series.size)])
        assert_array_equal(self.to_supervised_ty_single, expected_values)

    def test_ty_multiple_values(self):
        expected_values = np.array([list(range(n - (self.tx + self.ty_multiple - 1), n + 1))
                                    for n in
                                    range(self.tx + (self.ty_multiple - 1), self.test_series.size)])
        assert_array_equal(self.to_supervised_ty_multiple.values, expected_values)

    def test_shape_keep_na(self):
        keep_na_test = series_to_supervised(self.test_series, n_in=self.tx,
                                            n_out=self.ty_single, dropnan=False)
        expected_shape = (self.test_series.shape[0], self.tx + 1)
        self.assertEqual(keep_na_test.shape, expected_shape)