
import numpy as np
import pandas as pd
import pywt
import math
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

from keras import regularizers
from keras.metrics import mae
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, MaxoutDense, LSTM, LeakyReLU, Input, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

from pandas.plotting import scatter_matrix

from xgboost import XGBRegressor as xg
from lightgbm import LGBMRegressor, LGBMClassifier

from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

import warnings ; warnings.filterwarnings('ignore')



# 데이터 

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0)



def get_sonar(i):
    z = train.iloc[]  # input_mat['sonaralt']: (1, 1501)
    return z

