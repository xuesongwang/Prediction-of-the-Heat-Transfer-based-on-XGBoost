"""
__author__: Xuesong Wang
    2019 - 07 - 03
    Data helper functions including:
    read data
    normalization: fit and transform data, transformation: transform data only
    split data (unused in main.py)
    cross validation split (10-Fold)
    customized error function: MSE, MAPE, and R^2
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
# import seaborn


def read_data(filename):
    data = pd.read_csv(filename,dtype='float64')
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x, y


def normalization(x, method="standardization"):
    # notice that we do not normalize labels, only features are used
    if method == "standardization":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    x_norm = scaler.fit_transform(x)
    x_norm_df = pd.DataFrame(data=x_norm,columns=x.columns.values)
    return x_norm_df, scaler


def transformation(scaler, x):
    return pd.DataFrame(scaler.transform(x), columns=x.columns.values)


def split_data(x, y):
    return train_test_split(x, y, train_size = 60, random_state= 0)


def cross_val_split(x, y):
    '''
    given x and y, return a list containing cross validation split results
    '''
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    data = [{"x_train": x.iloc[train_index, :], "y_train":y.iloc[train_index],"x_val":x.iloc[val_index,:], "y_val":y.iloc[val_index]}
            for train_index, val_index in cv.split(x)]  # list comprehension
    return data


def rmse_fun(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape_fun(y_true, y_pred):
    return np.mean(np.abs(1.0*(y_pred - y_true)/y_true))


def r2_fun(y_true, y_pred):
    return 1 - np.mean((y_true - y_pred) ** 2)/np.mean(1.0*(y_true - np.mean(y_true))**2)


def overall_score(y_true, y_pred):
    rmse = rmse_fun(y_true, y_pred)
    mape = mape_fun(y_true, y_pred)
    r2 = r2_fun(y_true, y_pred)
    return [rmse, mape, r2]


def plot_coefficient_for_XGB(result,method):
    """
    plot Y_pred - Y_true for XGB regressor
    :param result: dict, format:
    {train: [y, y_pred], val: [y, y_pred], test: [y, y_pred]}
    :return: None, save figure
    """
    y_train = result["train"][0]
    y_train_pred= result["train"][1]
    y_val = result["val"][0]
    y_val_pred = result["val"][1]
    y_test = result["test"][0]
    y_test_pred = result["test"][1]
    fig, ax = plt.subplots()
    # set label size
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    # plot training set
    l1 = ax.scatter(y_train, y_train_pred, marker='^', s=150, c='blue', edgecolor = 'k', alpha=0.5)
    # plot validation set
    l2 = ax.scatter(y_val, y_val_pred, marker='s', s=120, c='red', edgecolor ='k', alpha=0.8)
    # plot testing set
    l3 = ax.scatter(y_test, y_test_pred, c='green', edgecolor = 'k', s=180, alpha=0.8)
    # plot the legend
    ax.legend([l1, l2, l3], labels=['training set', 'validation set', 'testing set'], loc='upper left', fontsize=15)
    # plot a diagnal line
    plt.plot([180, 1800], [180, 1800], 'k-', alpha=0.8)
    # plot axis limits
    ax.set_xlim(180, 1800)
    ax.set_ylim(180, 1800)
    # set x and y label content, using Latex
    ax.set_xlabel(r"Experimental $h$$_{\rm eff}$ (W/m$^{2}$K)", fontsize = 20, fontname = 'Times New Roman')
    ax.set_ylabel(r"Predicted $h$$_{\rm eff}$ (W/m$^{2}$K)",fontsize = 20, fontname = 'Times New Roman')
    plt.tight_layout()
    plt.savefig('./coefficient/'+method+"_coefficient.jpg", dpi=600)
    plt.show()


def generate_data_grid(x):
    """
    calculate x_median, fix other 8 features and interporlate within the rest one,
    predict values on this grid to see how a particular feature can affect the prediction
    :param x: training data
    :return: grids , a list containing generated data set, each data set fix 8 features and only change the rest one
    """
    x_grids = []
    x_median = np.median(x, axis=0)
    for feature in range(x.shape[1]):
        num = 100
        f_min = np.min(x.iloc[:, feature])
        f_max = np.max(x.iloc[:, feature])
        x_grid = x_median + np.zeros((num, 1))# use broadcasting to transform x_median to 2D, similar to np.repeat
        x_grid[:, feature] = np.linspace(f_min, f_max, num)
        x_grids.append(pd.DataFrame(x_grid, columns=x.columns.values))
    return x_grids