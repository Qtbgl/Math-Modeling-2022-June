import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
import warnings
warnings.filterwarnings('ignore')

#1导入数据
datasets = pd.read_csv('site_arima.csv')
dataset = datasets.iloc[:, 1].values.reshape(-1, 1)

#2切片
step_size = 15  # time_step = 15
data_input = np.zeros((len(dataset) - step_size, step_size))
for i in range(len(dataset)-step_size):
    data_input[i, :] = dataset[i:step_size + i, 0]
data_label = dataset[step_size:, 0]

#3划分数据集
test_number = 10
##训练集
X_train = data_input[:-test_number]
Y_train = data_label[:-test_number]
#测试集
X_test = data_input[-test_number:]
Y_test = data_label[-test_number:]

#4搭建预测模型
xgb = XGBRegressor(booster='gbtree',max_depth=40, learning_rate=0.2,reg_alpha=0.01, n_estimators=2000, gamma=0.1, min_child_weight=1)
xgb.fit(X_train,Y_train)

pre = xgb.predict(X_test)
# predict = xgb.predict(t5,t6,t7)


#5指标重要性可视化
importance = xgb.feature_importances_
plt.figure(1)
plt.barh(y = range(importance.shape[0]),  #指定条形图y轴的刻度值
         width = importance,  #指定条形图x轴的数值
         tick_label =range(importance.shape[0]),  #指定条形图y轴的刻度标签
         color = 'orangered',  #指定条形图的填充色
         )
plt.title('Feature importances of XGBoost')
#6计算评价指标
print(' MAE : ', mae(Y_test, pre))
print(' MAPE : ',mape(Y_test, pre))
print(' RMSE : ', np.sqrt(mse(Y_test, pre)))
#7结果可视化
plt.figure(2)
plt.plot(pre, color='red',label='predict')
plt.plot(Y_test, color='blue',label='true')
plt.title('Result visualization')
plt.legend()
plt.show()