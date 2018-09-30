import numpy as np
import datetime
from src.data.get_data import retrieve_all_data
from src.features.build_features import *
from src.CryptoPredict.Model import Model
from src.CryptoPredict.Preprocesser import Preprocesser

from xgboost import XGBRegressor

# The solution
if __name__ == '__main__':
    np.random.seed(31337)

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

    preprocessor = Preprocesser(data, TARGET, Tx, Ty, MOVING_AVERAGE_LAGS, name='Unit_Test')
    X, y, n_features = preprocessor.preprocess_train()

    print('Feature Matrix X Sample: {}'.format(X.sample(1, random_state=0).values[0][0]))
    print('Target Values y Sample: {}'.format(y.sample(1, random_state=0).values[0][0]))


    X_train, X_test, y_train, y_test = ttsplit_and_trim(X, y, TEST_SIZE, n_features, Ty)

    print('Train Feature Matrix X Sample: {}'.format(X_train.sample(1, random_state=0).values[0][0]))
    print('Train Target Values y Sample: {}'.format(y_train.sample(1, random_state=0).values[0][0]))
    print('Test Feature Matrix X Sample: {}'.format(X_test.sample(1, random_state=0).values[0][0]))
    print('Test Target Values y Sample: {}'.format(y_test.sample(1, random_state=0).values[0][0]))

    ta = Model(XGBRegressor(), 'xgboost_regressor')

    parameters = {
        'objective': 'reg:linear',
        'learning_rate': .07,
        'max_depth': 10,
        'min_child_weight': 4,
        'silent': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'n_estimators': 20
    }

    ta.set_parameters(parameters)

    train_scores = ta.fit(X_train, y_train)
    print('Train RMSE: {}'.format(train_scores[0]))
    print("Train MAE: {}\n".format(train_scores[1]))

    test_scores = ta.predict(X_test, y_test)
    print('Test RMSE: {}'.format(test_scores[0]))
    print("Test MAE: {}\n".format(test_scores[1]))