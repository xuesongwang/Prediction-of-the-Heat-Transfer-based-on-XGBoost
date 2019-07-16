"""
__author__: Xuesong Wang, Ning Qian
2019 - 07 -06
this file is to train a artificial neural network model using Keras,
prerequisite packages: tensorflow, Keras
"""

from DataProcessing import *
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model


class ANN:
    def __init__(self, input_dim = 1):
        self.input = Input(shape=(input_dim,))
        self.output = self.feed_forward()
        self.model = Model(inputs=self. input, outputs=self.output)
        self.model.compile(optimizer="adam", loss="mse", metrics=["mape"])
        self.model.summary()
    
    def feed_forward(self):
        """
        a 3-layer neural network, input dim: 9, output dim: 1
        layer structure: 9 -> 25 -> 8 -> 1
        activation function: last layer: no activation, the other layers: "relu" (to avoid gradient vanishing)
        batch normalization after each activation layer to accelerate training process
        :return: the predicted output
        """
        X = BatchNormalization()(self.input)
        X = Dense(25, activation="relu")(X)
        X = BatchNormalization()(X)
        X = Dense(8, activation="relu")(X)
        X = BatchNormalization()(X)
        X = Dense(1)(X)
        return X

    def fit(self, x, y, test_data):
        self.model.fit(x, y, validation_data=test_data, epochs=6000, batch_size=x.shape[0])

    def predict(self, x):
        return self.model.predict(x)


if __name__ == '__main__':
    trainfile = "../train_data.csv"
    testfile = "../test_data.csv"
    train_x, train_y = read_data(trainfile)
    test_x, test_y = read_data(testfile)
    train_val_data = cross_val_split(train_x, train_y) # split data into 10 folds
    train_x = train_val_data[9]["x_train"]
    train_y = train_val_data[9]["y_train"]
    val_x = train_val_data[9]["x_val"]
    val_y = train_val_data[9]["y_val"]
    model = ANN(input_dim=train_x.shape[1])
    model.fit(train_x, train_y, test_data=(val_x, val_y))
    train_y_pred = model.predict(x=train_x)
    val_y_pred = model.predict(x = val_x)
    test_y_pred = model.predict(x=test_x)
    train_score = overall_score(y_true=train_y, y_pred=train_y_pred)
    val_score = overall_score(y_true=val_y, y_pred =val_y_pred)
    test_score = overall_score(y_true=test_y, y_pred=test_y_pred)
    print ("train RMSE: %.4f, MAPE: %.2f%%, R^2: %.4f" % (train_score[0], train_score[1] * 100, train_score[2]))
    print ("val RMSE: %.4f, MAPE: %.2f%%, R^2: %.4f" % (val_score[0], val_score[1] * 100, val_score[2]))
    print ("test RMSE: %.4f, MAPE: %.2f%%, R^2: %.4f" % (test_score[0], test_score[1] * 100, test_score[2]))