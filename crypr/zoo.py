"""Welcome to the Zoo. All the different models - for your viewing!"""
from keras import Model
from keras.layers import Input, LSTM, BatchNormalization, Dense, Conv1D, Activation, Dropout, Reshape
from keras.regularizers import l1


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

    X = LSTM(units=128, return_sequences=False, kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization(axis=-1)(X)
    X = Dropout(0.5)(X)

    X = Dense(num_outputs, kernel_initializer=kernel_init, bias_initializer=bias_init, activation='linear')(X)

    return Model(inputs=X_input, outputs=X)


def LSTM_WSAEs(tx, num_channels=1, num_outputs=1, encoding_dim=10, kernel_init='normal', bias_init='zeros') -> Model:
    """Stacked Autoencoder + LSTM from Paper - WORK IN PROGRESS
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
        - Made up hidden_dim (only specified encoding_dim=10
    """

    # Encoder/Decoder hidden unit size halfway between num_inputs and encoding_dim
    hidden_dim = tx - int((tx - encoding_dim) / 2)

    # Autoencoder
    X_input = Input(shape=(tx, num_channels), dtype='float32')
    encoder = Dense(hidden_dim, activation='relu', name='encoder')(X_input)
    encoded = Dense(encoding_dim, activation='relu', name='encoded')(encoder)
    decoder = Dense(hidden_dim, activation='relu', name='decoder')(encoded)
    decoded = Dense(units=tx, activation='linear', name='decoded')(decoder)

    # LSTM
    X = Reshape(target_shape=(encoding_dim, num_channels), name='rs_0')(encoded)
    X = LSTM(units=64, return_sequences=False, kernel_initializer=kernel_init, bias_initializer=bias_init, name='lstm_0')(X)
    X = BatchNormalization(axis=-1, name='bn_0')(X)

    X = Dense(num_outputs, kernel_initializer=kernel_init, bias_initializer=bias_init, activation='linear', name='dense_0')(X)

    return Model(inputs=[X_input], outputs=[decoded, X])


def build_ae_lstm(tx, num_channels=1, num_outputs=1, kernel_init='normal', bias_init='zeros') -> Model:
    """Own implementation of a stacked autoencoder with LSTM for smoothed time-series data.
    Notes:
        - encoded layer is a sparse encoder (l1 regularized)
    """
    hidden_dim = int(tx / 2)
    encoding_dim = int(hidden_dim / 2)

    X_input = Input(shape=(num_channels, tx), dtype='float32', name='input_0')
    # Encoder
    encoder = Dense(units=hidden_dim, activation='relu', name='encoder')(X_input)
    encoded = Dense(units=encoding_dim, activation='relu', activity_regularizer=l1(1e-6), name='encoded')(encoder)
    # Decoder
    decoder = Dense(units=hidden_dim, activation='relu', name='decoder')(encoded)
    decoded = Dense(units=tx, activation='linear', name='decoded')(decoder)
    # LSTM One
    X = Reshape(target_shape=(encoding_dim, num_channels), name='rs_0')(encoded)
    X = LSTM(units=64, return_sequences=True, kernel_initializer=kernel_init, bias_initializer=bias_init, name='lstm_0')(X)
    X = Dropout(0.2, name='dr_0')(X)
    X = BatchNormalization(axis=-1, name='bn_0')(X)
    # LSTM Two
    X = LSTM(units=64, return_sequences=False, kernel_initializer=kernel_init, bias_initializer=bias_init, name='lstm_1')(X)
    X = Dropout(0.2, name='dr_1')(X)
    X = BatchNormalization(axis=-1, name='bn_1')(X)
    X = Dropout(0.2, name='dr_2')(X)

    # Output
    X = Dense(num_outputs, kernel_initializer=kernel_init, bias_initializer=bias_init, activation='linear', name='dense_0')(X)

    return Model(inputs=[X_input], outputs=[decoded, X])


