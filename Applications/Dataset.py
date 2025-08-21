import numpy as np

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class SPAM_import:
    """
    sets up SPAM dataset from OpenML 
    """
    def __init__(self):
        
        df = pd.read_csv("data/spamdata.csv", sep=' ') 
        
        # Create arrays for the features and the response variable
        # store for use later 
        df_shuffled = df.sample(frac = 1, random_state = 42).reset_index(drop = True)
        feature_name = df_shuffled.columns.tolist()
        y = df_shuffled['isSPAM'].values
        X = df_shuffled.drop('isSPAM', axis=1).values 

        # check row count and column count 
        print("Shape of y : ", y.shape) # Shape of y :  (70000,)
        print("Shape of X : ", X.shape) # Shape of X :  (70000, 784)

        # Convert the labels to numeric labels
        # is a function in the pandas library designed to convert an argument 
        # (such as a Series, list, or 1-d array) to a numeric type. 
        # It is particularly useful when dealing with data that may contain non-numeric values 
        # or values stored as strings that should be treated as numbers.
        y = np.array(pd.to_numeric(y))
        print("y to_numeric : ", y)

        test_split = 0.1
        train_size = int((1 - test_split) * len(df_shuffled))

        # avoid RuntimeWarning: overflow encountered in exp return 1 / (1 + np.exp(-z))
        # 特徵標準化（Standardization）
        # 若輸入 x_train 的值本身就非常大（例如原始 pixel 值、金錢、次方），會造成 𝑧=𝑤⋅𝑥+𝑏 z=w⋅x+b 絕對值變得很大
        # 使用 Z-score 標準化（均值為 0，標準差為 1）
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)

        # create training and validation sets 
        self.train_x, self.train_y = x_scaled[:train_size,:], y[:train_size]
        self.val_x, self.val_y = x_scaled[train_size:,:], y[train_size:]

class MNIST_import:
    """
    sets up MNIST dataset from OpenML 
    """
    def __init__(self):
        
        df = pd.read_csv("data/mnist_784.csv") # this is a 28*28 picture data set
        
        # Create arrays for the features and the response variable
        # store for use later 
        y = df['class'].values
        X = df.drop('class', axis=1).values # axis is an enum, 0: row, 1: column
        
        # check row count and column count 
        print("Shape of y : ", y.shape) # Shape of y :  (70000,)
        print("Shape of X : ", X.shape) # Shape of X :  (70000, 784)

        # Convert the labels to numeric labels
        # is a function in the pandas library designed to convert an argument 
        # (such as a Series, list, or 1-d array) to a numeric type. 
        # It is particularly useful when dealing with data that may contain non-numeric values 
        # or values stored as strings that should be treated as numbers.
        y = np.array(pd.to_numeric(y))
        print("y to_numeric : ", y)

        # avoid overflow
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)

        # create training and validation sets 
        self.train_x, self.train_y = x_scaled[:5000,:], y[:5000]
        self.val_x, self.val_y = x_scaled[5000:6000,:], y[5000:6000]

class IRIS_import: 
    def __init__(self):
        iris = datasets.load_iris()
        # print(type(iris)) # <class 'sklearn.utils._bunch.Bunch'>

        # check data type
        # self.train_x = iris.data[:, [0, 2]]
        # self.train_y = iris.target
        # print(type(self.train_x)) # <class 'numpy.ndarray'>

        # check data convert ndarray -> dataframe
        # X = pd.DataFrame(iris['data'], columns = iris['feature_names'])
        # y = pd.DataFrame(iris['target'], columns = ['target'])

        ## check 
        # iris_data = pd.concat([X, y], axis = 1)
        # iris_data = iris_data[['sepal length (cm)','petal length (cm)', 'target' ]]
        # iris_data = iris_data[iris_data['target'].isin([0,1])]
        # iris_data.head (3)
        # print(type(iris_data)) # <class 'pandas.core.frame.DataFrame'>
        # self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(iris_data[['sepal length (cm)','petal length (cm)']], iris_data[['target']], test_size = 0.2, random_state = 0)
        # print(type(self.train_x)) # <class 'pandas.core.frame.DataFrame'>

        # print(type(iris.target))
        # print(type(iris.data))
        # print(type(iris.data[['sepal length (cm)','petal length (cm)']]))

        #################
        feature_names = iris.feature_names
        idx_sepal_length = feature_names.index('sepal length (cm)')
        idx_petal_length = feature_names.index('petal length (cm)')

        X_selected = iris.data[:, [idx_sepal_length,idx_petal_length]]
        # print(type(X_selected)) # <class 'numpy.ndarray'>
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(X_selected, iris.target, test_size = 0.2, random_state = 0)
        # print(type(self.train_x)) # <class 'numpy.ndarray'>

        # avoid overflow 執行特徵縮放
        scaler = StandardScaler()
        self.train_x = scaler.fit_transform(self.train_x)
        self.val_x = scaler.fit_transform(self.val_x)
