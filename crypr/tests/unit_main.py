from sklearn.model_selection import train_test_split
import datetime
from crypr.data.get_data import retrieve_all_data
from crypr.features.build_features import *
from crypr.base.models import RegressionModel, SavedRegressionModel
from crypr.base.preprocessors import SimplePreprocessor

from xgboost import XGBRegressor

# The solution
if __name__ == '__main__':
    np.random.seed(31337)

    SYM = 'ETH'
    LAST_N_HOURS = 16000
    FEATURE_WINDOW=72
    MOVING_AVERAGE_LAGS = [6, 12, 24, 48, 72]
    TARGET = 'close'
    Tx = 72
    Ty = 1
    TEST_SIZE = 0.05

    data = retrieve_all_data(coin=SYM, num_hours=LAST_N_HOURS, comparison_symbol='USD',
                             end_time=(np.datetime64(datetime.datetime(2018, 6, 27)).astype('uint64') / 1e6).astype(
                                 'uint32'))

    preprocessor = SimplePreprocessor(production=False, target_col=TARGET, Tx=Tx, Ty=Ty, moving_averages=MOVING_AVERAGE_LAGS, name='unit_test')
    X, y = preprocessor.fit(data).transform(data)

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

    ta = RegressionModel(XGBRegressor(), 'xgboost_regressor')

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

    ta.estimator.set_params(**parameters)

    ta.fit(X_train, y_train)
    train_pred = ta.predict(X_train, y_train)
    train_rmse, train_mae = ta.evaluate(y_pred=train_pred, y_true=y_train)
    print('Train RMSE: {}'.format(train_rmse))
    print("Train MAE: {}\n".format(train_mae))

    ta = SavedRegressionModel('./tests/unit_xgboost_ETH_tx72_ty1_flag72.pkl')

    new_data = retrieve_all_data(SYM, Tx + FEATURE_WINDOW - 1,
                                 end_time=(np.datetime64(datetime.datetime(2018, 6, 27)).astype('uint64') / 1e6).astype(
                                 'uint32'))
    preprocessor = SimplePreprocessor(production=True, target_col=TARGET, Tx=Tx, Ty=Ty, moving_averages=[6, 12, 24, 48, 72],
                                name='Unit_New_Prediction_Preprocessor')
    X_new = preprocessor.fit(new_data).transform(new_data)
    prediction = ta.predict(X_new)[0]
    print('Prediction: {}'.format(prediction))