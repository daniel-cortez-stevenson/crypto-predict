"""Solution script for other tests"""
from os.path import join
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from crypr.cryptocompare import retrieve_all_data
from crypr.models import RegressionModel, SavedPickleRegressionModel
from crypr.preprocessors import SimplePreprocessor
from crypr.util import get_project_path, utc_timestamp_ymd


if __name__ == '__main__':
    np.random.seed(31337)

    SYM = 'ETH'
    LAST_N_HOURS = 14000
    FEATURE_WINDOW = 72
    MOVING_AVERAGE_LAGS = [6, 12, 24, 48, 72]
    TO_TIME = utc_timestamp_ymd(2018, 6, 27)
    TARGET = 'close'
    Tx = 72
    Ty = 1
    TEST_SIZE = 0.05

    project_path = get_project_path()
    test_data_dir = join(project_path, 'crypr', 'tests', 'data')

    data = retrieve_all_data(coin=SYM, num_hours=LAST_N_HOURS, comparison_symbol='USD', end_time=TO_TIME)

    preprocessor = SimplePreprocessor(production=False, target_col=TARGET, Tx=Tx, Ty=Ty,
                                      moving_averages=MOVING_AVERAGE_LAGS)
    X, y = preprocessor.fit(data).transform(data)

    old_shape = X.shape
    new_shape = (old_shape[0], old_shape[1] * old_shape[2])
    X = pd.DataFrame(np.reshape(a=X, newshape=new_shape), columns=preprocessor.engineered_columns)

    print('X shape: {}'.format(X.shape))
    print('y shape: {}'.format(y.shape))

    print('Feature Matrix X Sample: {}'.format(X.sample(1, random_state=0).values[0][0]))
    print('Target Values y Sample: {}'.format(y.sample(1, random_state=0).values[0][0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    print('y Train shape: {}'.format(y_train.shape))
    print('y Test shape: {}'.format(y_test.shape))

    print('Train Feature Matrix X Sample: {}'.format(X_train.sample(1, random_state=0).values[0][0]))
    print('Train Target Values y Sample: {}'.format(y_train.sample(1, random_state=0).values[0][0]))
    print('Test Feature Matrix X Sample: {}'.format(X_test.sample(1, random_state=0).values[0][0]))
    print('Test Target Values y Sample: {}'.format(y_test.sample(1, random_state=0).values[0][0]))

    ta = RegressionModel(XGBRegressor())

    parameters = {
        'objective': 'reg:linear',
        'learning_rate': .07,
        'max_depth': 10,
        'min_child_weight': 4,
        'silent': 1,
        'subsample': .7,
        'colsample_bytree': .7,
        'n_estimators': 20,
    }

    ta.estimator.set_params(**parameters)

    ta.fit(X_train, y_train)

    train_rmse, train_mae = ta.evaluate(X_pred=X_train, y_true=y_train)
    print('Train RMSE: {}'.format(train_rmse))
    print('Train MAE: {}\n'.format(train_mae))

    ta_model_filename = 'unit_xgboost_ETH_tx72_ty1_flag72.pkl'
    ta = SavedPickleRegressionModel(join(test_data_dir, ta_model_filename))

    new_data = retrieve_all_data(coin=SYM, num_hours=Tx + FEATURE_WINDOW - 1, end_time=TO_TIME)

    preprocessor = SimplePreprocessor(production=True, target_col=TARGET, Tx=Tx, Ty=Ty,
                                      moving_averages=MOVING_AVERAGE_LAGS)
    X_new = preprocessor.fit(new_data).transform(new_data)

    old_shape = X_new.shape
    new_shape = (old_shape[0], old_shape[1] * old_shape[2])
    X_new = pd.DataFrame(np.reshape(a=X_new, newshape=new_shape), columns=preprocessor.engineered_columns)

    prediction = ta.predict(X_new)[0]
    print('Prediction: {}'.format(prediction))
