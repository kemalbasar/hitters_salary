import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


def grab_col_name(df):
    cat_vars = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if
                   (str(df[col].dtypes) in ["int64", "float64"]) & (df[col].nunique() < 10)]
    cat_vars = cat_vars + num_but_cat
    num_vars = [col for col in df.columns if col not in cat_vars]
    cat_but_car = [col for col in df.columns if
                   (str(df[col].dtypes) in ["category", "object", "bool"]) & (df[col].nunique() > 20)]

    len_of_carvars = len(cat_but_car)
    len_of_numvars = len(num_vars)
    len_of_catvars = len(cat_vars)

    print('Categoric Vars = %d ' % len_of_catvars)
    print('Numeric Vars = %d ' % len_of_numvars)
    print('Cardinal Vars = %d ' % len_of_carvars)

    print('Categoric Vars: ' + str(cat_vars))
    print('Numeric Vars: ' + str(num_vars))
    print('Cardinal Vars: ' + str(cat_but_car))

    return cat_vars, num_vars, cat_but_car


def cat_summary(df, column, plot=False, targetvar=False):
    """
    It describe the spesifications of variable.

    Parameters
    ----------
    df: Dataframe
        Dataframe you want to analyse
    column: string
        the column which you will get summary of
    plot: bool
        parameter which decide plot gonne show or not
    targetvar: string
        name of target variable analyse


    Returns
    -------

    """
    print(pd.DataFrame({column: df[column].value_counts(),
                        'rate': df[column].value_counts() / len(df)}))
    print("###############################################")

    if targetvar:
        print(df.pivot_table(targetvar, column))

    if plot:
        sns.countplot(x=df[column], data=df)
        plt.show(block=True)


def num_summary(dataframe, column, plot=False, targetvar=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[column].describe(quantiles).T)

    if targetvar:
        print(dataframe.pivot_table(targetvar, column))

    if plot:
        dataframe[column].hist()
        plt.xlabel(column)
        plt.title(column)
        plt.show(block=True)


def get_core_triangle(df):
    core = df.corr()
    core_triangle = core.where(np.triu(np.ones(core.shape), k=1).astype(np.bool_))
    return core_triangle




def binary_encoder(df):
    binary_col = [col for col in df.columns if df[col].dtype not in [int, float]
                   and df[col].nunique() == 2]

    for col in binary_col:
        labelencoder = LabelEncoder()
        df[col] = labelencoder.fit_transform(df[col])
    # labelencoder.inverse_transform([])
    return df



