# # Installation
# ! apt-get update
get_ipython().system(u' apt-get install -y --allow-unauthenticated swig')
get_ipython().system(u' pip install pyrfr')
get_ipython().system(u' pip install Cython')
get_ipython().system(u' pip install auto-sklearn')
# # Hello World
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import autosklearn.regression
import numpy as np
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')
def main():
    X, y = sklearn.datasets.load_boston(return_X_y=True)
    feature_types = (['numerical'] * 3) + ['categorical'] + (['numerical'] * 9)
    X_train, X_test, y_train, y_test =         sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=36,
        per_run_time_limit=30
    )
    automl.fit(X_train, y_train, dataset_name='boston', feat_type=feature_types)
    print(automl.show_models())
    predictions = automl.predict(X_test)
    print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))
    print("MAE score:", sklearn.metrics.mean_absolute_error(y_test, predictions))
    print("RMSE score:", np.sqrt(sklearn.metrics.mean_squared_error(y_test, predictions)))
    dump(automl, 'model.joblib')
main()
def test_joblib():
    X, y = sklearn.datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = load('model.joblib')
    print(automl.predict(X_test))
    return automl
model = test_joblib()
