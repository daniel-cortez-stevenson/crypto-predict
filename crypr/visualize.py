"""Visualize Model Performance"""
import matplotlib.pyplot as plt


def learning_curve(fitted_keras_model, loss_train='loss', loss_val='val_loss'):
    """Check out loss from train and dev sets"""
    plt.figure()
    plt.plot(fitted_keras_model.history[loss_train], label='train')
    plt.plot(fitted_keras_model.history[loss_val], label='val')
    plt.legend()
    plt.show()
    return None
