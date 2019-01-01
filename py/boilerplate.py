# # RUN MODE
DEV_MODE = True
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import ElasticNet, Lasso, LassoCV
# Comment this if the data visualisations doesn't work on your side
get_ipython().magic(u'matplotlib inline')
plt.style.use('bmh')
pd.options.display.max_columns = 100
sns.set()
if DEV_MODE:
    from sklearn.datasets import load_boston
    boston = load_boston()
    boston_df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
# # EDA Library
# ## heatMap(df, abs_cor = 0.4)
def heatMap(df, abs_cor=0.4):
    corrmat = df.corr()
    plt.subplots(figsize=(15, 12))
    sns.set(font_scale=1)
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corrmat[(corrmat > abs_cor) | (corrmat < -abs_cor)],
        vmax=.8,
        square=True,
        annot=True,
        mask=mask,
        cmap=cmap
    )
if DEV_MODE:
    heatMap(boston_df)
# ## getNA(df)
def getNA(dataframe):
    columns = []
    nas = []
    types = []
    at_least_one = False
    for column in list(dataframe):
        hasNA = dataframe[[column]].isna().sum().values
        if (hasNA[0] > 0):
            columns = columns + [column]
            nas = nas + [hasNA[0]]
            types = types + [dataframe[column].dtype]
            at_least_one = True
    if at_least_one == False:
        return 'No NA found'
    return pd.DataFrame({'columns': columns, 'na': nas, 'dtype': types})
if DEV_MODE:
    print(getNA(boston_df))
    boston_df['na'] = None
    print(getNA(boston_df))
def getColumnData(df, column):
    if type(column) == int:
        return df.columns[column]
    elif type(column) == str:
        return df[column]
    return column
# ## singleFieldAnalysis(column, df = None)
def singleFieldAnalysis(column, df=None):
    column = getColumnData(df, column)
    print(column.describe())
    print("Na percent:", column.isna().sum() / len(column))
    if column.dtype != 'object':
        print("Skewness:", column.dropna().skew())
        print("Kurtosis:", column.dropna().kurt())
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        sns.distplot(column.dropna())
        plt.subplot(1, 2, 2)
        stats.probplot(column, plot=plt)
        plt.show()
    else:
        temp = column.fillna('~none')
        temp = sns.countplot(
            y=temp, order=temp.value_counts().index, color='#bc5090'
        )
if DEV_MODE:
    singleFieldAnalysis('AGE', boston_df)
    singleFieldAnalysis(np.log1p(boston_df['AGE']))
# ## scatter(y, x1, x2 = None, x3 = None, x4 = None, df = None)
def scatter(y, x1, x2=None, x3=None, x4=None, df=None):
    x1_data = getColumnData(df, x1)
    x2_data = getColumnData(df, x2)
    x3_data = getColumnData(df, x3)
    x4_data = getColumnData(df, x4)
    y_data = getColumnData(df, y)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(x=x1_data, y=y_data)
    if df is not None:
        plt.ylabel(y)
        plt.xlabel(x1)
    if x2 is not None:
        plt.subplot(2, 2, 2)
        plt.scatter(x=x2_data, y=y_data)
        if df is not None:
            plt.ylabel(y)
            plt.xlabel(x2)
    if x3 is not None:
        plt.subplot(2, 2, 3)
        plt.scatter(x=x3_data, y=y_data)
        if df is not None:
            plt.ylabel(y)
            plt.xlabel(x3)
    if x4 is not None:
        plt.subplot(2, 2, 4)
        plt.scatter(x=x4_data, y=y_data)
        if df is not None:
            plt.ylabel(y)
            plt.xlabel(x4)
if DEV_MODE:
    print(scatter('AGE', 'NOX', df=boston_df))
# # ML Library
# ## doXgbCV(model, train, test, metrics, cv_folds = 5, early_stopping_rounds = 50)
def doXgbCV(model, train, test, metrics, cv_folds=5, early_stopping_rounds=50):
    xgb_param = model.get_xgb_params()
    xgtrain = xgb.DMatrix(train.values, label=test.values)
    cvresult = xgb.cv(
        xgb_param,
        xgtrain,
        num_boost_round=model.get_params()['n_estimators'],
        nfold=cv_folds,
        metrics=metrics,
        early_stopping_rounds=early_stopping_rounds
    )
    print(cvresult[['test-rmse-mean', 'test-rmse-std']].tail(1))
    return cvresult
def doSklearnCV(
    model, train, test, metrics="neg_mean_squared_error", cv_folds=5
):
    kf = KFold(
        cv_folds, shuffle=True, random_state=42
    ).get_n_splits(train.values)
    print(train.shape)
    print(test.shape)
    rmse = np.sqrt(
        -cross_val_score(
            model, train.values, test.values, scoring=metrics, cv=kf
        )
    )
    print("Score: {:.4f} ({:.4f})".format(rmse.mean(), rmse.std()))
    return (rmse, model)
# ## doLassoCV(train, test, metrics="neg_mean_squared_error", cv_folds = 5, alpha=1)
def doLassoCV(
    train, test, metrics="neg_mean_squared_error", cv_folds=5, alpha=1
):
    return doSklearnCV(Lasso(alpha=alpha), train, test, metrics, cv_folds)
# ## lassoFeatureImportance(train, test,  cv=5)
def lassoFeatureImportance(train, test, cv=5):
    model = LassoCV(alphas=[1, 0.1, 0.001, 0.0005], cv=cv)
    model.fit(train, test)
    coef = pd.Series(model.coef_, index=train.columns)
    print('Best alpha:', model.alpha_)
    print('Removed features:', coef[coef == 0].index)
    coef[coef != 0].sort_values().plot(kind="barh")
    plt.title("Coefficients in the Lasso Model")
if DEV_MODE:
    ignore = lassoFeatureImportance(boston_df.fillna(0), boston.target)
# # Misc Library
# ## doGc()
def doGc():
    gc.collect()
    get_ipython().system(u' free -m')
get_ipython().system(u' free -m')
