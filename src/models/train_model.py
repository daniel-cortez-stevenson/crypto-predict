from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam

def make_train_lr(X, Y):
    lr_model = LinearRegression()
    lr_model = lr_model.fit(X, Y)
    return lr_model


def make_lstm(input_shape, num_outputs):
    model = Sequential()

    model.add(LSTM(64, input_shape=input_shape, activation='linear', return_sequences=True))

    model.add(Dropout(rate=0.1))

    model.add(LSTM(64, activation='linear'))

    model.add(Dense(num_outputs, activation='linear'))

    return model


def train_lstm(model, X_train, Y_train, X_test, Y_test, num_epochs=20, batch_size=128):
    model.compile(loss='mae', optimizer=Adam(lr=0.0001))

    fit = model.fit(X_train, Y_train,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, Y_test))
    return model, fit