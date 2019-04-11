from unittest import TestCase
from os.path import join
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from crypr.util import get_project_path
from crypr.transformers import BaseTransformer, PassthroughTransformer, PercentChangeTransformer, MovingAverageTransformer


class TestTransformer(TestCase):
    def setUp(self):
        np.random.seed(31337)
        self.decimal_tolerance = 10

        self.test_cols = ['high', 'low', 'close']
        self.moving_average_lag = 4

        self.test_data_path = join(get_project_path(), 'crypr', 'tests', 'data', 'test_raw_btc.csv')
        self.data = pd.read_csv(self.test_data_path, index_col=0)

    def test_transformer_get_feature_names(self):
        transformers = [('transformer name', BaseTransformer(), self.test_cols)]
        ct = ColumnTransformer(transformers=transformers)
        ct.fit(self.data)
        self.assertListEqual(['transformer name__' + col for col in self.test_cols], ct.get_feature_names())

    def test_passthrough_transformer(self):
        transformers = [('passthrough', PassthroughTransformer(), self.test_cols)]
        ct = ColumnTransformer(transformers=transformers)
        data_transformed = ct.fit_transform(X=self.data)
        self.assertIsNone(
            np.testing.assert_array_equal(x=data_transformed, y=self.data[self.test_cols].values)
        )

    def test_percent_change_transformer(self):
        transformers = [('percent_change', PercentChangeTransformer(), self.test_cols)]
        ct = ColumnTransformer(transformers=transformers)
        data_transformed = ct.fit_transform(X=self.data)
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                x=data_transformed,
                y=(self.data[self.test_cols].pct_change()*100.).values,
                decimal=self.decimal_tolerance,
            )
        )

    def test_moving_average_transformer(self):
        transformers = [('moving_average', MovingAverageTransformer(n=self.moving_average_lag), self.test_cols)]
        ct = ColumnTransformer(transformers=transformers)
        data_transformed = ct.fit_transform(X=self.data)
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                x=data_transformed,
                y=self.data[self.test_cols].rolling(self.moving_average_lag).mean().values,
                decimal=self.decimal_tolerance,
            )
        )

