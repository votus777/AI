
'''

m30에 _eval2 와 eval3 만들것

1. eval 예 'loss' 와 다른 지표 1개 더 추가 
2. earlystopping 적용
3. plot 으로 그릴 것 

SelectfromModel 에 
1. 회귀                             m29_eval1_SFM
2. 이진 분류                        m29_eval2_SFM
3. 다중 분류                        m29_eval3_SFM


4. 결과는 주석으로 소스 하단에 표시 

5. m27 ~ 29까지 완벽 이해할 것 


'''


from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

dataset = load_boston()

x, y = load_boston(return_X_y=True)

print(x.shape)  
print(y.shape) 

x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 16)


model = XGBRegressor( n_estimators = 300 , learning_rate = 0.1)

model.fit(x_train, y_train, verbose = True, eval_metric = "rmse",  
                                            eval_set = [(x_train,y_train), (x_test,y_test)], 
                                            early_stopping_rounds = 20)


 # eval_metric => rmse, mae, logloss, error, auc, roc 
 
results = model.evals_result()

# print("eval's result : ", results)


y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)

print("R2 score : %.2ff%%" %(r2*100.0))

'''
[1]     validation_0-rmse:19.60139      validation_1-rmse:17.86865
[2]     validation_0-rmse:17.74823      validation_1-rmse:16.11803
[3]     validation_0-rmse:16.07556      validation_1-rmse:14.47598
[4]     validation_0-rmse:14.56910      validation_1-rmse:13.12597
[5]     validation_0-rmse:13.22035      validation_1-rmse:11.87671
[6]     validation_0-rmse:11.99750      validation_1-rmse:10.69085
[7]     validation_0-rmse:10.90463      validation_1-rmse:9.73972
[8]     validation_0-rmse:9.91748       validation_1-rmse:8.90409
[9]     validation_0-rmse:9.02793       validation_1-rmse:8.08961
[10]    validation_0-rmse:8.21866       validation_1-rmse:7.31619
[11]    validation_0-rmse:7.49246       validation_1-rmse:6.63007
[12]    validation_0-rmse:6.83416       validation_1-rmse:6.07082
[13]    validation_0-rmse:6.24288       validation_1-rmse:5.57650
[14]    validation_0-rmse:5.71072       validation_1-rmse:5.11737
[15]    validation_0-rmse:5.22838       validation_1-rmse:4.70913
[16]    validation_0-rmse:4.79804       validation_1-rmse:4.36621
[17]    validation_0-rmse:4.40932       validation_1-rmse:4.04894
[18]    validation_0-rmse:4.06144       validation_1-rmse:3.79225
[19]    validation_0-rmse:3.74144       validation_1-rmse:3.53170
[20]    validation_0-rmse:3.45181       validation_1-rmse:3.35497
[21]    validation_0-rmse:3.19646       validation_1-rmse:3.19287
[22]    validation_0-rmse:2.96322       validation_1-rmse:3.07765
[23]    validation_0-rmse:2.75391       validation_1-rmse:2.91854
[24]    validation_0-rmse:2.56740       validation_1-rmse:2.80173
[25]    validation_0-rmse:2.40023       validation_1-rmse:2.72147
[26]    validation_0-rmse:2.24726       validation_1-rmse:2.65751
[27]    validation_0-rmse:2.11155       validation_1-rmse:2.60116
[28]    validation_0-rmse:1.99341       validation_1-rmse:2.56599
[29]    validation_0-rmse:1.88189       validation_1-rmse:2.54935
[30]    validation_0-rmse:1.78686       validation_1-rmse:2.53064
[31]    validation_0-rmse:1.70084       validation_1-rmse:2.52026
[32]    validation_0-rmse:1.62648       validation_1-rmse:2.52039
[33]    validation_0-rmse:1.55425       validation_1-rmse:2.52258
[34]    validation_0-rmse:1.48259       validation_1-rmse:2.52073
[35]    validation_0-rmse:1.42534       validation_1-rmse:2.51752
[36]    validation_0-rmse:1.37555       validation_1-rmse:2.51214
[37]    validation_0-rmse:1.32730       validation_1-rmse:2.51439
[38]    validation_0-rmse:1.28535       validation_1-rmse:2.52596
[39]    validation_0-rmse:1.24108       validation_1-rmse:2.54289
[40]    validation_0-rmse:1.21089       validation_1-rmse:2.55562
[41]    validation_0-rmse:1.18000       validation_1-rmse:2.57401
[42]    validation_0-rmse:1.15042       validation_1-rmse:2.57301
[43]    validation_0-rmse:1.11892       validation_1-rmse:2.58764
[44]    validation_0-rmse:1.09292       validation_1-rmse:2.59329
[45]    validation_0-rmse:1.07500       validation_1-rmse:2.59794
[46]    validation_0-rmse:1.04320       validation_1-rmse:2.59227
[47]    validation_0-rmse:1.01821       validation_1-rmse:2.59871
[48]    validation_0-rmse:0.99705       validation_1-rmse:2.60973
[49]    validation_0-rmse:0.97757       validation_1-rmse:2.64344
[50]    validation_0-rmse:0.96521       validation_1-rmse:2.64409
[51]    validation_0-rmse:0.92775       validation_1-rmse:2.67020
[52]    validation_0-rmse:0.90710       validation_1-rmse:2.67488
[53]    validation_0-rmse:0.89658       validation_1-rmse:2.67866
[54]    validation_0-rmse:0.88308       validation_1-rmse:2.68923
[55]    validation_0-rmse:0.86957       validation_1-rmse:2.69362
[56]    validation_0-rmse:0.85060       validation_1-rmse:2.69260
Stopping. Best iteration:
[36]    validation_0-rmse:1.37555       validation_1-rmse:2.51214

R2 score : 91.01f%
'''