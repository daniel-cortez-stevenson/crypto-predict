"""Welcome to the Zoo. All the different models - for your viewing!"""
from keras import Model
from keras.layers import Input, LSTM, BatchNormalization, Dense, Conv1D, Activation, Dropout, Flatten


def LSTM_triggerNG(tx, num_channels, num_outputs, kernel_init='normal', bias_init='zeros') -> Model:
    """Modified from Andrew Ng's Deeplearning.ai Sequence Modeling course on Coursera.
    Originally used for trigger word detection.
    Notes:
        - Switched GRU cells to LSTM cells
        - BiDirectional???
        - Custom, guesswork kernel_size and strides in Conv1D
    """
    X_input = Input(shape=(tx, num_channels), dtype='float32')

    X = Conv1D(196, kernel_size=8, strides=2, kernel_initializer=kernel_init, bias_initializer=bias_init)(X_input)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)

    X = LSTM(units=128, return_sequences=True, kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization(axis=-1)(X)

    X = LSTM(units=128, return_sequences=True, kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization(axis=-1)(X)
    X = Dropout(0.5)(X)

    X = Flatten()(X)
    X = Dense(num_outputs, kernel_initializer=kernel_init, bias_initializer=bias_init, activation='linear')(X)

    return Model(inputs=X_input, outputs=X)
