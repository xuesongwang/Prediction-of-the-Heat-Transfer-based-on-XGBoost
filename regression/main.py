'''
__author__: Xuesong Wang, Ning Qian
    2019 - 07 -01
    this file is to compare the following regressors using the optimal parameters from main_grid_search:
    linear regression,
    support vector regression
    random forest, xgboost, lightgbm
'''

from DataProcessing import *
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance, plot_tree
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt


class Classifier:
    def __init__(self, name, cv=1):
        '''
        initilize  a list of regressors with the length equaling cv,
        the initialization parameters were obtained from main_grid_research
        :param name:  regressor name
        :param cv: the number of cross folds for validation
        '''
        self.name = name
        if name == "LR":
            self.clfs = [LinearRegression() for _ in range(cv)]
        elif name == "SVR":
            self.clfs = [SVR(C=1e-3) for _ in range(cv)]
        elif name == "RF":
            self.clfs = [RandomForestRegressor(n_estimators=106, min_samples_leaf=1,
                                               min_samples_split=4, max_depth=42,
                                               max_features=4, random_state=1,
                                               n_jobs=4) for _ in range(cv)]
        elif name == "XGB":
            self.clfs = [XGBRegressor(learning_rate=0.19, n_jobs=4,
                                      n_estimators=580, max_depth=5,
                                      subsample=0.8, colsample_bytree=0.8,
                                      gamma=0, random_state=1) for _ in range(cv)]
        elif name == "LGB":
            self.clfs = [LGBMRegressor(n_estimators=580, learning_rate=0.19, num_iterations=1000,
                                       boosting_type='dart', min_data_in_leaf=1,
                                       max_depth=6, num_leaves=20) for _ in range(cv)]
            
    def fit(self, data):
        '''
        fit all the models in self.clfs using cross fold data
        :param data: list of cross fold data, format: e.g., data[0]["x_train"], data[9]["y_val"]
        :return:training and validation score on cross fold data
        '''
        train_RMSE = []
        train_MAPE = []
        train_R2 = []
        val_RMSE = []
        val_MAPE =[]
        val_R2 = []
        best_cv = 0  # the index of the validation set achieving the lowest error
        best_val_score = 1e5
        for cv, cv_data in enumerate(data):
            x_train = cv_data["x_train"]
            y_train = cv_data["y_train"]
            x_val = cv_data["x_val"]
            y_val = cv_data["y_val"]
            self.clfs[cv].fit(x_train, y_train)
            print ("current clf:",cv)
            train_score = overall_score(y_train, self.clfs[cv].predict(x_train))
            train_RMSE.append(train_score[0])
            train_MAPE.append(train_score[1])
            train_R2.append(train_score[2])
            val_score = overall_score(y_val, self.clfs[cv].predict(x_val))
            val_RMSE.append(val_score[0])
            val_MAPE.append(val_score[1])
            val_R2.append(val_score[2])
            if val_score[0] < best_val_score:
                best_cv = cv
        self.best_estimator = self.clfs[best_cv]  #use the best clf with the lowest validation error
        self.best_cv = best_cv
        if self.name == "XGB":
            self.plot_estimator("importance")
        score = {"train_RMSE": np.mean(train_RMSE), "train_MAPE":np.mean(train_MAPE), "train_R2": np.mean(train_R2),
                 "val_RMSE": np.mean(val_RMSE), "val_MAPE": np.mean(val_MAPE), "val_R2": np.mean(val_R2)}
        return score

    def score(self,x, y):
        """
        regression scores on (x,y) using self.best_estimator,
        predict
        :param x:  test_x
        :param y:  test_y
        :return:  test_score
        """
        y_pred = self.predict(x)
        RMSE = rmse_fun(y, y_pred)
        MAPE = mape_fun(y, y_pred)
        R2 = r2_fun(y, y_pred)
        score = {"RMSE": RMSE, "MAPE": MAPE, "R2": R2}
        return score

    def predict(self, x):
        '''
        predict output using self.best_estimator given input x
        :param x: input x
        :return: predicted y
        '''
        return self.best_estimator.predict(x)

    def plot_estimator(self,item):
        """
        plot critical information of an XGB regressor
        if item == "importance", plot feature importance
        if item == "tree", plot the tree structure of listed in tree_index
        :param item:  "importance" or "tree"
        :return: None, save plots
        """
        if item == "importance":
            plot_importance(self.best_estimator,importance_type="gain", xlabel="Feature contribution",ylabel=None,title=None,grid=False,xlim=(0,35000))
            plt.tight_layout()
            plt.savefig("feature_importance_by_gain.jpg", dpi=600)
            plt.show()
        else:
            tree_index = [0, 1, 100, 300, 400] # tree index used to plot structure
            for index in tree_index:
                fig, ax = plt.subplots()
                fig.set_size_inches(300,150)
                plot_tree(self.best_estimator, num_trees=index, ax=ax)
                plt.savefig('./tree structure/tree_' + str(index)+".jpg")
                plt.show()

    def plot_coefficient(self,cv_data, test_data):
        """
        plot Prediction-Experiment coefficients for train,val and test data
        :param cv_data: train and val data
        :param test_data: test data
        :return: plot figure
        """
        y_train_pred = self.predict(cv_data["x_train"])
        y_val_pred = self.predict(cv_data["x_val"])
        y_test_pred = self.predict(test_data[0])
        result = {"train": [cv_data["y_train"], y_train_pred],
                  "val": [cv_data["y_val"], y_val_pred],
                  "test": [test_data[1], y_test_pred]
                  }
        plot_coefficient_for_XGB(result, method=self.name)


if __name__ == '__main__':
    trainfile = "../train_data.csv"
    testfile = "../test_data.csv"
    train_x, train_y = read_data(trainfile)
    test_x, test_y = read_data(testfile)
    model_list = ["LR", "SVR", "RF", "XGB", "LGB"]
    model_index = 3 # model_index = 3, means that we are choosing XGB
    if model_index >= 2:  # do not need to normalize data for tree based models
        train_x_norm = train_x
        test_x_norm = test_x
    else:
        train_x_norm, scaler = normalization(train_x, method="standardization")
        test_x_norm = transformation(scaler, test_x)
    train_val_data = cross_val_split(train_x_norm, train_y) # split data into 10 folds
    model = Classifier(model_list[model_index], cv=len(train_val_data)) # model initialization
    score = model.fit(train_val_data)
    test_score = model.score(test_x_norm, test_y)
    best_cv = model.best_cv
    print ("train RMSE: %.4f, MAPE: %.2f%%, R^2: %.4f"%(score["train_RMSE"], score["train_MAPE"]*100, score["train_R2"]))
    print ("val RMSE: %.4f, MAPE: %.2f%%, R^2: %.4f" % (score["val_RMSE"], score["val_MAPE"]*100, score["val_R2"]))
    print ("test RMSE: %.4f, MAPE: %.2f%%, R^2: %.4f" % (test_score["RMSE"], test_score["MAPE"]*100, test_score["R2"]))

    # plot correlation correfient for training, testing and validation
    model.plot_coefficient(train_val_data[best_cv], test_data=(test_x_norm, test_y))

    x_grids = generate_data_grid(train_val_data[best_cv]["x_train"])
    for index, x_grid in enumerate(x_grids):
        y_grid = model.predict(x_grid)
        fig,ax = plt.subplots()
        plt.plot(x_grid.iloc[:, index], y_grid, 'k-')
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_xlabel(x_grid.columns.values[index], fontsize=15, fontname = "Times New Roman")
        ax.set_ylabel(r"Predicted $h$$_{\rm eff}$ (W/m$^{2}$K)", fontsize=20)
        plt.tight_layout()
        plt.savefig('./decision boundary/'+ model_list[model_index] + '_' +(x_grid.columns.values[index]).replace('/', '-')+'.jpg', dpi=600)
        plt.show()
