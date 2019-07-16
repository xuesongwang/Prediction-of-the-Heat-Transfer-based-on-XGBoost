"""
__author__: Xuesong Wang, Ning Qian
    2019 - 07 -01
    this file is to use grid search to find the optimal parameters for the following regressors :
    linear regression,
    support vector regression
    random forest, xgboost, lightgbm
"""

from DataProcessing import *
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt


class Classifier:
    def __init__(self, name):
        if name == "LR":
            self.clf = LinearRegression()
        elif name == "SVR":
            self.clf = SVR()
            self.tuned_param = {"kernel":['rbf'],             # parameters to be tuned and define a searching space
                                "C": [ 1e-3, 5e-3, 0.01]}
        elif name == "RF":
            self.clf = RandomForestRegressor(n_jobs=4)
            self.tuned_param = {
                                "n_estimators": range(95,110),
                                "criterion":['mse'],
                                "max_depth": [42],
                                "min_samples_split": [4],
                                "min_samples_leaf": [1],
                                "max_features": [4],
                                }
        elif name == "XGB":
            self.clf = XGBRegressor(n_jobs=4)
            self.tuned_param = {
                                "n_estimators": range(500, 650, 20),
                                "learning_rate": np.arange(0.01, 0.25, 0.05),
                                "max_depth": range(3,7,2),
                                "min_child_weight": range(3, 7, 2),
                                "subsample": [0.8],
                                "colsample_bytree": [0.8],
                                "gamma": np.arange(0, 0.6, 0.3),
                                "reg_alpha": np.arange(0.01, 0.15, 0.05) # regularization
                                }
        elif name == "LGB":
            self.clf = LGBMRegressor(num_iterations = 400, n_jobs=4)
            self.tuned_param = {
                                "boosting_type":['dart'],
                                "learning_rate": [0.03],
                                "max_depth": [6],
                                "num_leaves": range(20,40),
                                "n_estimators": [100],
                                "min_data_in_leaf": [1]
                                }

    def plot_grid_search(self, x, y):
        """
        tune only one parameter at a time, fix other params, plot the validation curve,
        from which speculate the optimal range of the best parameters
        :param x: use the whole training data (GridSearchCV will automatically use cross validation)
        :param y:  the whole training labels
        :return: None, save plots for further examination
        """
        def evaluate_param(parameter, num_range, index):
            grid_search = GridSearchCV(self.clf, param_grid={parameter: num_range},cv=10,
                                       scoring={"RMSE": make_scorer(rmse_fun),
                                                "MAPE": make_scorer(mape_fun),
                                                "R2": make_scorer(r2_fun)},
                                       refit="RMSE", return_train_score=True
                                       )
            grid_search.fit(x, y)
            plot = plt.plot(num_range, grid_search.cv_results_["mean_test_RMSE"])
            plt.title(parameter)
            plt.savefig(parameter + ".jpg")
            plt.show()
            return plot
        index = 0
        for parameter, param_range in dict.items(self.tuned_param):
            evaluate_param(parameter, param_range, index)
            index += 1

    def grid_search(self, x, y):
        """
        tune all parameter space, the optimal range of all the parameters have defined a
        parameter grid, return the best parameters as well as the best regressor
        :param x: the whole training data
        :param y:  the whole training label
        :return: train and validation score on cross fold data
        """
        grid_clf = GridSearchCV(self.clf,self.tuned_param,cv = 10,
                                scoring={"RMSE": make_scorer(rmse_fun),
                                         "MAPE": make_scorer(mape_fun),
                                         "R2": make_scorer(r2_fun)},
                                refit="RMSE",return_train_score=True)
        result = grid_clf.fit(x, y)
        print ("best parameters", result.best_params_)
        self.clf = result.best_estimator_
        score = {"train_RMSE": np.mean(result.cv_results_["mean_train_RMSE"]),
                 "train_MAPE": np.mean(result.cv_results_["mean_train_MAPE"]),
                 "train_R2": np.mean(result.cv_results_["mean_train_R2"]),
                 "val_RMSE": np.mean(result.cv_results_["mean_test_RMSE"]),
                 "val_MAPE": np.mean(result.cv_results_["mean_test_MAPE"]),
                 "val_R2": np.mean(result.cv_results_["mean_test_R2"])
                 }
        return score


if __name__ == '__main__':
    trainfile = "../train_data.csv"
    testfile = "../test_data.csv"
    train_x, train_y = read_data(trainfile)
    test_x, test_y = read_data(testfile)
    namelist = ["LR", "SVR", "RF", "XGB", "LGB"]
    model_index = 3
    model = Classifier(namelist[model_index])
    if model_index >= 2:  # do not need to normalize data for tree based models
        train_x_norm = train_x
        test_x_norm = test_x
    else:
        train_x_norm, scaler = normalization(train_x, method="standardization")
        test_x_norm = transformation(scaler, test_x)
    plot_only = False
    if plot_only == True:
        # plot grid search is to use plot to find best parameter range
        model.plot_grid_search(train_x_norm, train_y)
    else:
        # grid search is to search the best parameter in that range
        score = model.grid_search(train_x_norm, train_y)
        print ("train RMSE: %.4f, val RMSE: %.4f"%(score["train_RMSE"], score["val_RMSE"]))
        print ("train MAPE: %.2f%%, val MAPE: %.2f%%" % (score["train_MAPE"]*100, score["val_MAPE"]*100))
        print ("train R^2: %.4f, val R^2: %.4f "%(score["train_R2"], score["val_R2"]))

