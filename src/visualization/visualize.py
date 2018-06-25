import matplotlib.pyplot as plt


def learning_curve(fitted_keras_model):
    """
    Check out loss from train and dev sets
    """
    plt.figure()
    plt.plot(fitted_keras_model.history['loss'], label='train')
    plt.plot(fitted_keras_model.history['val_loss'], label='test')
    plt.legend()
    plt.show()