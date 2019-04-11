from unittest import TestCase
from numpy.testing import assert_array_equal
from os.path import join
import numpy as np
import pandas as pd
from crypr.util import get_project_path
from crypr.build import keep_last_n_rows, make_3d, series_to_supervised, strip_nan_rows


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


class TestHelperFunctions(TestCase):
    def setUp(self):
        np.random.seed(27772)
        self.tx = 3
        self.num_channels = 3
        self.num_samples = 10
        self.data_1d = np.arange(start=0, stop=self.tx * self.num_channels * self.num_samples, dtype='float')
        self.data_2d = self.data_1d.reshape((self.num_samples, self.num_channels * self.tx))
        self.data_3d = self.data_2d.reshape((self.num_samples, self.tx, self.num_channels))

    def test_strip_nan_rows(self):
        # insert nan into arr
        nan_data = self.data_2d.copy()
        num_nan_rows = 3
        nan_row_ixs = np.random.choice(self.num_samples, size=num_nan_rows, replace=False)
        nan_col_ix = np.random.choice(self.tx * self.num_channels, size=1, replace=False)[0]
        nan_data[nan_row_ixs, nan_col_ix] = np.nan
        # strip nan rows (test assertion)
        nan_drop_data = strip_nan_rows(nan_data)
        # delete nan manually
        deleted_data = np.delete(arr=self.data_2d, obj=nan_row_ixs, axis=0)

        np.testing.assert_array_equal(x=deleted_data, y=nan_drop_data)

    def test_keep_last_n_rows(self):
        test_n = 4
        last_n_rows_arr = keep_last_n_rows(arr=self.data_3d, n=test_n)
        np.testing.assert_array_equal(x=self.data_3d[(self.num_samples - test_n):, :, :], y=last_n_rows_arr)

    def test_make_3d(self):
        test_3d_arr = make_3d(self.data_3d, tx=self.tx, num_channels=self.num_channels)
        np.testing.assert_array_equal(x=self.data_3d, y=test_3d_arr)
