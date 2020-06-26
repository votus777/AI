import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib

from statsmodels.tsa.arima_model import ARIMA

from keras.models import Sequential 
from keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten 

from xgboost import XGBRegressor, plot_importance
from lightgbm import LGBMRegressor, LGBMClassifier 

train = pd.read_csv('./AI_2020/train.csv', header = 1, index_col = [0,1])
val = pd.read_csv('./AI_2020/val.csv', header = 1, index_col = [0,1])
test = pd.read_csv('./AI_2020/test.csv', header = 1, index_col = [0,1])


train.index.freq = 'MS'
ax = train['경부선'].plot(figsize = (180,50))
ax.set(xlabel='Dates', ylabel='Total');

plt.figure(figsize = (16,7))
train.plot()
        
plt.show()

'''
 #   Column      Non-Null Count  Dtype        
---  ------      --------------  -----        
 0   시간          2895 non-null   int64      
 1   경부선         2895 non-null   int64     
 2   남해선(순천-부산)  2895 non-null   int64
 3   남해선(영암-순천)  2895 non-null   int64
 4   광주대구선       2895 non-null   int64
 5   무안광주선       2895 non-null   int64
 6   고창담양선       2895 non-null   int64
 7   서해안선        2895 non-null   int64
 8   울산선         2895 non-null   int64
 9   대구포항선       2895 non-null   int64
 10  익산장수선       2895 non-null   int64
 11  호남선         2895 non-null   int64
 12  순천완주선       2895 non-null   int64
 13  청주영덕선       2895 non-null   int64
 14  당진대전선       2895 non-null   int64
 15  통영대전선       2895 non-null   int64
 16  중부선         2895 non-null   int64
 17  제2중부선       2895 non-null   int64
 18  평택제천선       2895 non-null   int64
 19  중부내륙선       2895 non-null   int64
 20  영동선         2895 non-null   int64
 21  중앙선         2895 non-null   int64
 22  서울양양선       2895 non-null   int64
 24  동해선(부산-포항)  2895 non-null   int64
 25  서울외곽순환선     2895 non-null   int64
 26  남해1지선       2895 non-null   int64
 27  남해2지선       2895 non-null   int64
 28  제2경인선       2895 non-null   int64
 29  경인선         2895 non-null   int64
 30  서천공주선       2895 non-null   int64
 31  호남지선        2895 non-null   int64
 32  대전남부선       2895 non-null   int64
 33  중부내륙지선      2895 non-null   int64
 34  중앙선지선       2895 non-null   int64
 35  부산외곽선       2895 non-null   int64
 '''




'''

192 88

print(train.columns)

def outliers(data_out):
    outliers = []
    for i in range(len(data_out.columns)):
        data = data_out.iloc[:, i]
        quartile_1 = data.quantile(.25)
        quartile_3 = data.quantile(.75)
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        out = np.where((data > upper_bound) | (data < lower_bound))
        outliers.append(out)
        print(i,'번째 outlier')
        
        
    return outliers



print(outliers(train))

train = np.array(train)

print(train.shape)

index = [(2125,2126),(3,26),(555,626),(2850,2873)]

train = np.delete(train, np.s_[3:27], axis=0)  
train = np.delete(train, np.s_[555:626], axis=0)  

print(train.shape)
'''




'''
1- (array([], dtype=int64),), 
2-  (array([], dtype=int64),), 
3- (array([], dtype=int64),), 
4-(array([ 562,  563,  564,  565,  566,  586,  587,  588,  589,  590,  591,
        592,  593,  610,  611,  612,  613,  614,  615,  616,  617, 2857,
       2858, 2862, 2863, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888,
       2889], dtype=int64),), 
5-(array([ 562,  563,  564,  565,  566,  586,  587,  588,  589,  590,  591,
        592,  593,  594,  610,  611,  612,  613,  614,  615,  616,  617,
       2857, 2858, 2859, 2860, 2882, 2883, 2885, 2886, 2887], dtype=int64),),
6-(array([588, 589, 590, 591], dtype=int64),),
7-(array([ 563,  586,  587,  588,  589,  590,  591,  592,  593,  611,  614,
        615,  616,  617, 2857, 2858, 2859, 2862, 2863, 2864, 2865, 2882,
       2885, 2886, 2887, 2888], dtype=int64),), 
8-(array([588, 590], dtype=int64),), 
9-(array([], dtype=int64),), 
10-(array([  11,   12,   13,   14,   15,  562,  563,  564,  587,  588,  589,
        590,  591,  592,  593,  611,  612,  613,  614,  615,  616,  617,
       2888], dtype=int64),),
11-(array([ 562,  563,  564,  565,  566,  586,  587,  588,  589,  590,  591,
        592,  593,  611,  612,  613,  614,  615,  616, 2857, 2858, 2884,
       2885, 2886], dtype=int64),), 
12-(array([587, 588, 589, 590, 591, 592], dtype=int64),), 
13-(array([ 562,  563,  566,  586,  587,  588,  589,  590,  591,  592,  593,
        611,  612,  613,  614,  615,  616,  617, 2857, 2858, 2859, 2860,
       2861, 2862, 2863, 2864, 2865, 2881, 2882, 2883, 2884, 2885, 2886,
       2887, 2888, 2889], dtype=int64),),
14-(array([  10,   11,   12,   13,   14,   15,   16,  111,  563,  567,  568,
        587,  588,  589,  590,  591,  592,  593,  594,  611,  612,  613,
        614,  615,  616,  617, 2857, 2858, 2859, 2860, 2861, 2862, 2863,
       2882, 2883, 2884, 2885, 2886, 2887, 2888], dtype=int64),),
15-(array([ 563,  564,  586,  587,  588,  589,  590,  591,  592,  593,  614,
        615,  616, 2857, 2858, 2859, 2862, 2863, 2882, 2886, 2887, 2888],
      dtype=int64),), 
16-(array([ 561,  562,  563,  564,  565,  566,  567,  586,  587,  588,  589,
        590,  591,  592,  593,  594,  610,  611,  612,  613,  614,  615,
        616,  617,  618,  638,  639, 2857, 2858, 2859, 2860, 2861, 2862,
       2863, 2864, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889],
      dtype=int64),),
 17-(array([], dtype=int64),)
 18-(array([], dtype=int64),),
19-(array([587], dtype=int64),),
20-(array([563, 587, 588, 589, 590, 591, 592, 593, 614], dtype=int64),),
21-(array([], dtype=int64),),
 22-(array([ 543,  544,  545,  560,  561,  562,  563,  564,  566,  586,  587,
        588,  589,  590,  591,  592,  593,  594,  595,  610,  611,  612,
        613,  614,  615,  616,  617, 2858, 2862, 2863, 2886, 2887, 2888],
      dtype=int64),), 
 23-(array([  12,   13,   83,   84,   85,  251,  252,  253,  255,  276,  277,
        278,  444,  445,  587,  588,  589,  590,  591,  592,  593,  610,
        611,  612,  613,  614,  615,  616, 1093, 1094, 1095, 1612, 1755,
       1756, 1757, 1777, 1778, 1779, 1923, 1924, 1925, 1926, 1946, 1947,
       2113, 2114, 2115, 2427, 2428, 2860, 2861, 2862, 2863, 2864, 2865,
       2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889], dtype=int64),),
24-(array([  11,   12,  563,  564,  587,  588,  589,  590,  591,  592,  593,
        610,  611,  612,  613,  614,  615,  616, 2861, 2862, 2863, 2864,
       2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889], dtype=int64),)
           , (array([563, 564, 587, 588, 589, 590, 591, 592, 593, 611, 612, 613, 614,
       615, 616, 617], dtype=int64),), 
  25-(array([], dtype=int64),),
 26-(array([587, 588, 589, 590, 591, 592, 611, 612], dtype=int64),),
  27-(array([], dtype=int64),),
 28-(array([], dtype=int64),), 
29-(array([], dtype=int64),), 
30-(array([ 543,  544,  545,  547,  548,  549,  550,  557,  559,  560,  561,
        562,  563,  564,  565,  566,  567,  568,  585,  586,  587,  588,
        589,  590,  591,  592,  593,  594,  595,  610,  611,  612,  613,
        614,  615,  616,  617,  618,  619,  620,  638, 2262, 2263, 2430,
       2431, 2597, 2598, 2599, 2765, 2766, 2767, 2856, 2857, 2858, 2859,
       2860, 2861, 2862, 2863, 2864, 2865, 2866, 2881, 2882, 2883, 2884,
       2885, 2886, 2887, 2888, 2889, 2890], dtype=int64),),
31-(array([], dtype=int64),)
32-(array([545, 546, 562, 563, 586, 587, 588, 589, 590, 591, 592, 593],dtype=int64),), 
33-(array([], dtype=int64),), 
34-(array([], dtype=int64),),
35-(array([], dtype=int64),)]

'''