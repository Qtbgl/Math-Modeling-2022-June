import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
import tensorflow as tf


# 导入数据
datasets = pd.read_csv('site_arima.csv')
dataset = datasets.iloc[:, 1].values.reshape(-1, 1)

# 归一化
scaled_tool = MinMaxScaler(feature_range=[0, 1])
data_scaled = scaled_tool.fit_transform(dataset)

# 切片
step_size = 15
data_seg = np.zeros((len(data_scaled) - step_size, step_size))
for i in range(len(data_scaled) - step_size):
    data_seg[i, :] = data_scaled[i: i + step_size, 0]
data_label = data_scaled[step_size:, 0]

# 数据集划分
test_number = 10
X_train = data_seg[: -test_number]
Y_train = data_label[: -test_number]
X_test = data_seg[-test_number:]
Y_test = data_label[-test_number:]

# 搭建预测模型
inputs = tf.keras.Input(shape=(15,))
x = tf.keras.layers.Dense(28, activation='relu', name='dense_1')(inputs)  # 28 = 15 * 2 - 2
x1 = tf.keras.layers.Dense(28, activation='relu', name='dense_2')(x)
x2 = tf.keras.layers.Dense(28, activation='relu', name='dense_3')(x1)
x3 = tf.keras.layers.Dense(28, activation='relu', name='dense_4')(x2)
x4 = tf.keras.layers.Dense(1, name='dense_5')(x3)
model = tf.keras.Model(inputs=inputs, outputs=x4)

# 配置模型优化器、损失函数
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='mean_squared_error')
# 训练模型
model.fit(X_train, Y_train, batch_size=40, epochs=100)

# 预测输出
Y_pre = model.predict(X_test)

# 反归一化
Y_pre = np.reshape(Y_pre, (X_test.shape[0], 1))
Y_test = np.reshape(Y_test, (X_test.shape[0], 1))
Y_pre = scaled_tool.inverse_transform(Y_pre)
Y_test = scaled_tool.inverse_transform(Y_test)

# 模型效果评价
print('test RMSE : %.3f' % np.sqrt(mse(Y_test, Y_pre)))
print('test MAE : %.3f' % mae(Y_test, Y_pre))
print('test R2 : %.3f' % r2(Y_test, Y_pre))

# 绘图
plt.figure(1)
plt.plot(Y_test)
plt.plot(Y_pre, color='red')
plt.show()