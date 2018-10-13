"""
Welcome to the Zoo. All the different models - for your viewing!
"""
from keras.layers import Input, LSTM, BatchNormalization, Dense, Conv1D, TimeDistributed, Activation
from keras import Model

"""
Modified from Andrew Ng's Deeplearning.ai Sequence Modeling course on Coursera.
Originally used for trigger word detection.
Alternative Archetecture w/ Conv1D
"""
def LSTM_triggerNG(input_shape, num_outputs, kernel_init='normal', bias_init='zeros'):

    model_input = Input(shape=input_shape, dtype='float32')

    X = Conv1D(128, kernel_size=4, strides=2, kernel_initializer=kernel_init, bias_initializer=bias_init)(model_input)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)

    X = LSTM(units=64, return_sequences=True, kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    X = BatchNormalization(axis=-1)(X)

    X = LSTM(units=64, return_sequences=False, kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    X = BatchNormalization(axis=-1)(X)

    X = Dense(num_outputs, kernel_initializer=kernel_init, bias_initializer=bias_init, activation='linear') \
        (X) # time distributed  (linear)

    model = Model(inputs=model_input, outputs=X)
    return model
