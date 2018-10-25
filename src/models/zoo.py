"""
Welcome to the Zoo. All the different models - for your viewing!
"""
from keras.layers import Input, LSTM, GRU, BatchNormalization, Dense, Conv1D, TimeDistributed, Activation, Dropout
from keras import Model


def LSTM_triggerNG(input_shape, num_outputs, kernel_init='normal', bias_init='zeros'):
    """
    Modified from Andrew Ng's Deeplearning.ai Sequence Modeling course on Coursera.
    Originally used for trigger word detection.
    Notes:
        - Switched GRU cells to LSTM cells
        - BiDirectional
        - Custom, guesswork kernel_size and strides in Conv1D
    """

    model_input = Input(shape=input_shape, dtype='float32')

    X = Conv1D(196, kernel_size=4, strides=2, kernel_initializer=kernel_init, bias_initializer=bias_init)(model_input)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)

    X = LSTM(units=128, return_sequences=True, go_backwards=True, kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization(axis=-1)(X)

    X = LSTM(units=128, return_sequences=False, go_backwards=True, kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization(axis=-1)(X)
    X = Dropout(0.8)(X)

    X = Dense(num_outputs, kernel_initializer=kernel_init, bias_initializer=bias_init, activation='linear')(X)

    model = Model(inputs=model_input, outputs=X)
    return model


def LSTM_WSAEs(input_shape, num_outputs, num_autoencoder=4, encoding_dim=10, kernel_init='normal', bias_init='zeros'):
    """
    WORK IN PROGRESS
    TODO:
        ?num_delays = 5
        ?stateful

    From Paper:
        "A deep learning framework for financial time series using stacked autoencoders and long-short term memory"
        Authors:
            Wei Bao,
            Jun Yue,
            Yulei Rao
    Notes:
        - Added BatchNormalization Layer after LSTM
        - Increased LSTM hidden unit layer to 64

    """

    model_input = Input(shape=input_shape, dtype='float32')
    X = Dense(encoding_dim, activation='relu', name='ae_0')(model_input)

    for enc_layer in range(num_autoencoder-1):
        X = Dense(encoding_dim, activation='relu', name='ae_{}'.format(enc_layer+1))(X)

    X = LSTM(units=64, return_sequences=False, kernel_initializer=kernel_init, bias_initializer=bias_init, name='lstm_0')(X)
    X = BatchNormalization(axis=-1, name='bn_0')(X)

    X = Dense(num_outputs, kernel_initializer=kernel_init, bias_initializer=bias_init, activation='linear', name='dense_0')(X)

    model = Model(inputs=model_input, outputs=X)
    return model