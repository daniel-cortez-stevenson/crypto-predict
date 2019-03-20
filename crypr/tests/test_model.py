import unittest
import numpy as np
from crypr.models import RegressionModel
from crypr.zoo import LSTM_WSAEs, build_ae_lstm


class TestModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(31337)

        batch_size = 32
        Tx = 72

        self.X = np.random.randint(low=0, high=100, size=(batch_size, 1, 72)) / 100
        self.y = np.random.randint(low=0, high=1000, size=(batch_size,))
        self.models = [
            RegressionModel(build_ae_lstm(Tx, 1, 1)),
            RegressionModel(LSTM_WSAEs(Tx, 1, 1)),
        ]
        self.model_inputs = [self.X, self.X]
        self.model_outputs = [
            [self.X, self.y],
            [self.X, self.y],
        ]
        self.fits = []

        # Train all models for 1 epoch
        for i in range(len(self.models)):
            model = self.models[i]
            model.estimator.compile(loss='mae', optimizer='adam')
            fit = model.fit(self.model_inputs[i], self.model_outputs[i], shuffle=False, epochs=1, batch_size=batch_size)
            self.fits.append(fit)

    def test_model_is_training(self):
        model_weights = [model.estimator.trainable_weights for model in self.models]
        for weights in model_weights:
            for var in weights:
                self.assertTrue(var.initial_value != var.value())

    def test_loss(self):
        model_losses = [np.array(fit.history['loss']) for fit in self.fits]
        for loss in model_losses:
            self.assertTrue((loss > 0).all())
