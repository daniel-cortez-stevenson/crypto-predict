import numpy as np
import datetime
from src.data.get_data import retrieve_all_data
from src.features.build_features import *
from src.CryptoPredict.Model import Model

# The solution
if __name__ == '__main__':
    SYM = 'ETH'
    LAST_N_HOURS = 16000
    MOVING_AVERAGE_LAGS = [6, 12, 24, 48, 72]
    TARGET = 'close'
    Tx = 72
    Ty = 1
    TEST_SIZE = 0.05

    data = retrieve_all_data(coin=SYM, num_hours=LAST_N_HOURS, comparison_symbol='USD',
                             end_time=(np.datetime64(datetime.datetime(2018, 6, 27)).astype('uint64') / 1e6).astype(
                                 'uint32'))

    df = data[['open', 'high', 'close', 'low', 'volumeto', 'volumefrom']] \
        .pipe(calc_target, TARGET) \
        .pipe(calc_volume_ma, MOVING_AVERAGE_LAGS) \
        .dropna(how='any', axis=0)

    N_FEATURES = len(df.columns)

    X, y = data_to_supervised(df, Tx, Ty)

    X_train, X_test, y_train, y_test = ttsplit_and_trim(X, y, TEST_SIZE, N_FEATURES, Ty)

    np.random.seed(31337)
    ta = Model(X_train, y_train, X_test, y_test)
    train_scores = ta.fit()
    print()
    print('Train RMSE:', train_scores[0], '\n')
    print("Train MAE:\n{}\n".format(train_scores[1]))

    test_scores = ta.predict()
    print()
    print('Test RMSE:', test_scores[0], '\n')
    print("Test MAE:\n{}\n".format(test_scores[1]))