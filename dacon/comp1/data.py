
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras import metrics
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, MaxoutDense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import KFold, cross_validate

# 데이터 

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0)

print(train.shape)

train = train.values

dst = train[ : , 1]
src = train[ : , 36]
print(dst)
print(src)




# rho = train.sort_values(by =['rho'], axis=0)
# rho.to_csv('./data/dacon/comp1/rho.csv', index_label='id')

# sns.pairplot(train, diag_kind='hist')
# plt.show()

# corr_matrix = train.corr()
# hbb = corr_matrix["na"].sort_values(ascending=True)
# print(hbb)

# sns.pairplot(train_dst, diag_kind='hist')


# import missingno as msno
# msno.matrix(train_dst)  # dst가 난리다 
# plt.show()

# train.hist(bins=50, figsize=(20,20))


# plt.figure(figsize=(4, 12))
# sns.heatmap(train.corr().loc['rho':'990_dst', 'hhb':].abs())

# test.filter(regex='_dst$',axis=1).tail().T.plot() 
# plt.show()


# train = train.drop('840_dst', axis=1)
# test = test.drop(['810_dst','830_dst','840_dst'], axis=1 )



'''
class AutoEncoder:
    def __init__(self, encoding_dim):
        self.encoding_dim = encoding_dim
    def build_train_model(self, input_shape, encoded1_shape, encoded2_shape, decoded1_shape, decoded2_shape):
        input_data = Input(shape=(1, input_shape))
        encoded1 = Dense(encoded1_shape, activation="relu", activity_regularizer=regularizers.l2(0))(input_data)
        encoded2 = Dense(encoded2_shape, activation="relu", activity_regularizer=regularizers.l2(0))(encoded1)
        encoded3 = Dense(self.encoding_dim, activation="relu", activity_regularizer=regularizers.l2(0))(encoded2)
        decoded1 = Dense(decoded1_shape, activation="relu", activity_regularizer=regularizers.l2(0))(encoded3)
        decoded2 = Dense(decoded2_shape, activation="relu", activity_regularizer=regularizers.l2(0))(decoded1)
        decoded = Dense(input_shape, activation="sigmoid", activity_regularizer=regularizers.l2(0))(decoded2)
        autoencoder = Model(inputs=input_data, outputs=decoded)
        encoder = Model(input_data, encoded3)
        # Now train the model using data we already preprocessed
        autoencoder.compile(loss="mae", optimizer="adam")
        train = pd.read_csv("preprocessing/rbm_train.csv", index_col=0)
        ntrain = np.array(train)
        train_data = np.reshape(ntrain, (len(ntrain), 1, input_shape))
        # print(train_data)
        # autoencoder.summary()
        autoencoder.fit(train_data, train_data, epochs=1000)


class NeuralNetwork:
    def __init__(self, input_shape, stock_or_return):
        self.input_shape = input_shape
        self.stock_or_return = stock_or_return
    def make_train_model(self):
        input_data = Input(shape=(1, self.input_shape))
        lstm = LSTM(5, input_shape=(1, self.input_shape), return_sequences=True, activity_regularizer=regularizers.l2(0.003),
                       recurrent_regularizer=regularizers.l2(0), dropout=0.2, recurrent_dropout=0.2)(input_data)
        perc = Dense(5, activation="sigmoid", activity_regularizer=regularizers.l2(0.005))(lstm)
        lstm2 = LSTM(2, activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001),
                        dropout=0.2, recurrent_dropout=0.2)(perc)
        out = Dense(1, activation="sigmoid", activity_regularizer=regularizers.l2(0.001))(lstm2)
        model = Model(input_data, out)
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
        # load data
        train = np.reshape(np.array(pd.read_csv("features/autoencoded_train_data.csv", index_col=0)),
                           (len(np.array(pd.read_csv("features/autoencoded_train_data.csv"))), 1, self.input_shape))
        train_y = np.array(pd.read_csv("features/autoencoded_train_y.csv", index_col=0))
        # train_stock = np.array(pd.read_csv("train_stock.csv"))
        # train model
        model.fit(train, train_y, epochs=2000)


alpha.loc[alpha['900_dst'].isnull(),'900_dst']=alpha.loc[alpha['900_dst'].isnull(),'910_dst']
alpha.loc[alpha['910_dst'].isnull(),'910_dst']=alpha.loc[alpha['910_dst'].isnull(),'920_dst']
alpha.loc[alpha['920_dst'].isnull(),'920_dst']=alpha.loc[alpha['920_dst'].isnull(),'930_dst']
alpha.loc[alpha['930_dst'].isnull(),'930_dst']=alpha.loc[alpha['930_dst'].isnull(),'940_dst']
alpha.loc[alpha['940_dst'].isnull(),'940_dst']=alpha.loc[alpha['940_dst'].isnull(),'950_dst']
alpha.loc[alpha['950_dst'].isnull(),'950_dst']=alpha.loc[alpha['950_dst'].isnull(),'960_dst']
alpha.loc[alpha['960_dst'].isnull(),'960_dst']=alpha.loc[alpha['960_dst'].isnull(),'970_dst']
alpha.loc[alpha['970_dst'].isnull(),'970_dst']=alpha.loc[alpha['970_dst'].isnull(),'980_dst']
alpha.loc[alpha['980_dst'].isnull(),'980_dst']=alpha.loc[alpha['980_dst'].isnull(),'990_dst']


alpha.loc[alpha['800_dst'].isnull(),'800_dst']=alpha.loc[alpha['800_dst'].isnull(),'810_dst']
alpha.loc[alpha['810_dst'].isnull(),'810_dst']=alpha.loc[alpha['810_dst'].isnull(),'820_dst']
alpha.loc[alpha['820_dst'].isnull(),'820_dst']=alpha.loc[alpha['820_dst'].isnull(),'830_dst']
alpha.loc[alpha['830_dst'].isnull(),'830_dst']=alpha.loc[alpha['830_dst'].isnull(),'840_dst']
alpha.loc[alpha['840_dst'].isnull(),'840_dst']=alpha.loc[alpha['840_dst'].isnull(),'850_dst']
alpha.loc[alpha['850_dst'].isnull(),'850_dst']=alpha.loc[alpha['850_dst'].isnull(),'860_dst']
alpha.loc[alpha['860_dst'].isnull(),'860_dst']=alpha.loc[alpha['860_dst'].isnull(),'870_dst']
alpha.loc[alpha['870_dst'].isnull(),'870_dst']=alpha.loc[alpha['870_dst'].isnull(),'880_dst']
alpha.loc[alpha['880_dst'].isnull(),'880_dst']=alpha.loc[alpha['880_dst'].isnull(),'890_dst']
alpha.loc[alpha['890_dst'].isnull(),'890_dst']=alpha.loc[alpha['890_dst'].isnull(),'900_dst']

alpha.loc[alpha['710_dst'].isnull(),'710_dst']=alpha.loc[alpha['710_dst'].isnull(),'720_dst']
alpha.loc[alpha['720_dst'].isnull(),'720_dst']=alpha.loc[alpha['720_dst'].isnull(),'730_dst']
alpha.loc[alpha['730_dst'].isnull(),'730_dst']=alpha.loc[alpha['730_dst'].isnull(),'740_dst']
alpha.loc[alpha['740_dst'].isnull(),'740_dst']=alpha.loc[alpha['740_dst'].isnull(),'750_dst']
alpha.loc[alpha['750_dst'].isnull(),'750_dst']=alpha.loc[alpha['750_dst'].isnull(),'760_dst']
alpha.loc[alpha['760_dst'].isnull(),'760_dst']=alpha.loc[alpha['760_dst'].isnull(),'770_dst']
alpha.loc[alpha['770_dst'].isnull(),'770_dst']=alpha.loc[alpha['770_dst'].isnull(),'780_dst']
alpha.loc[alpha['780_dst'].isnull(),'780_dst']=alpha.loc[alpha['780_dst'].isnull(),'790_dst']
alpha.loc[alpha['790_dst'].isnull(),'790_dst']=alpha.loc[alpha['790_dst'].isnull(),'800_dst']

alpha.loc[alpha['700_dst'].isnull(),'700_dst']=alpha.loc[alpha['700_dst'].isnull(),'710_dst']
alpha.loc[alpha['690_dst'].isnull(),'690_dst']=alpha.loc[alpha['690_dst'].isnull(),'700_dst']
alpha.loc[alpha['680_dst'].isnull(),'680_dst']=alpha.loc[alpha['680_dst'].isnull(),'690_dst']
alpha.loc[alpha['670_dst'].isnull(),'670_dst']=alpha.loc[alpha['670_dst'].isnull(),'680_dst']
alpha.loc[alpha['660_dst'].isnull(),'660_dst']=alpha.loc[alpha['660_dst'].isnull(),'670_dst']
alpha.loc[alpha['650_dst'].isnull(),'650_dst']=alpha.loc[alpha['650_dst'].isnull(),'660_dst']

#======================================================================================================

beta.loc[beta['900_dst'].isnull(),'900_dst']=beta.loc[beta['900_dst'].isnull(),'910_dst']
beta.loc[beta['910_dst'].isnull(),'910_dst']=beta.loc[beta['910_dst'].isnull(),'920_dst']
beta.loc[beta['920_dst'].isnull(),'920_dst']=beta.loc[beta['920_dst'].isnull(),'930_dst']
beta.loc[beta['930_dst'].isnull(),'930_dst']=beta.loc[beta['930_dst'].isnull(),'940_dst']
beta.loc[beta['940_dst'].isnull(),'940_dst']=beta.loc[beta['940_dst'].isnull(),'950_dst']
beta.loc[beta['950_dst'].isnull(),'950_dst']=beta.loc[beta['950_dst'].isnull(),'960_dst']
beta.loc[beta['960_dst'].isnull(),'960_dst']=beta.loc[beta['960_dst'].isnull(),'970_dst']
beta.loc[beta['970_dst'].isnull(),'970_dst']=beta.loc[beta['970_dst'].isnull(),'980_dst']
beta.loc[beta['980_dst'].isnull(),'980_dst']=beta.loc[beta['980_dst'].isnull(),'990_dst']

beta.loc[beta['800_dst'].isnull(),'800_dst']=beta.loc[beta['800_dst'].isnull(),'810_dst']
beta.loc[beta['810_dst'].isnull(),'810_dst']=beta.loc[beta['810_dst'].isnull(),'820_dst']
beta.loc[beta['820_dst'].isnull(),'820_dst']=beta.loc[beta['820_dst'].isnull(),'830_dst']
beta.loc[beta['830_dst'].isnull(),'830_dst']=beta.loc[beta['830_dst'].isnull(),'840_dst']
beta.loc[beta['840_dst'].isnull(),'840_dst']=beta.loc[beta['840_dst'].isnull(),'850_dst']
beta.loc[beta['850_dst'].isnull(),'850_dst']=beta.loc[beta['850_dst'].isnull(),'860_dst']
beta.loc[beta['860_dst'].isnull(),'860_dst']=beta.loc[beta['860_dst'].isnull(),'870_dst']
beta.loc[beta['870_dst'].isnull(),'870_dst']=beta.loc[beta['870_dst'].isnull(),'880_dst']
beta.loc[beta['880_dst'].isnull(),'880_dst']=beta.loc[beta['880_dst'].isnull(),'890_dst']
beta.loc[beta['890_dst'].isnull(),'890_dst']=beta.loc[beta['890_dst'].isnull(),'900_dst']

beta.loc[beta['710_dst'].isnull(),'710_dst']=beta.loc[beta['710_dst'].isnull(),'720_dst']
beta.loc[beta['720_dst'].isnull(),'720_dst']=beta.loc[beta['720_dst'].isnull(),'730_dst']
beta.loc[beta['730_dst'].isnull(),'730_dst']=beta.loc[beta['730_dst'].isnull(),'740_dst']
beta.loc[beta['740_dst'].isnull(),'740_dst']=beta.loc[beta['740_dst'].isnull(),'750_dst']
beta.loc[beta['750_dst'].isnull(),'750_dst']=beta.loc[beta['750_dst'].isnull(),'760_dst']
beta.loc[beta['760_dst'].isnull(),'760_dst']=beta.loc[beta['760_dst'].isnull(),'770_dst']
beta.loc[beta['770_dst'].isnull(),'770_dst']=beta.loc[beta['770_dst'].isnull(),'780_dst']
beta.loc[beta['780_dst'].isnull(),'780_dst']=beta.loc[beta['780_dst'].isnull(),'790_dst']
beta.loc[beta['790_dst'].isnull(),'790_dst']=beta.loc[beta['790_dst'].isnull(),'800_dst']

beta.loc[beta['700_dst'].isnull(),'700_dst']=beta.loc[beta['700_dst'].isnull(),'710_dst']
beta.loc[beta['690_dst'].isnull(),'690_dst']=beta.loc[beta['690_dst'].isnull(),'700_dst']
beta.loc[beta['680_dst'].isnull(),'680_dst']=beta.loc[beta['680_dst'].isnull(),'690_dst']
beta.loc[beta['670_dst'].isnull(),'670_dst']=beta.loc[beta['670_dst'].isnull(),'680_dst']
beta.loc[beta['660_dst'].isnull(),'660_dst']=beta.loc[beta['660_dst'].isnull(),'670_dst']
beta.loc[beta['650_dst'].isnull(),'650_dst']=beta.loc[beta['650_dst'].isnull(),'660_dst']

'''