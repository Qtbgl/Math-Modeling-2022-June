import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def generate_purchase_seq():
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    df = pd.read_csv('site_arima.csv', parse_dates=['time'],
                               index_col='time', date_parser=dateparse)

    seq_train = df['2018-1-1':'2018-12-23']
    seq_test = df['2018-12-24':'2018-12-31']
    return seq_train, seq_test

def diff(timeseries):
    timeseries_diff1 = timeseries.diff(1)
    timeseries_diff2 = timeseries_diff1.diff(1)

    timeseries_diff1 = timeseries_diff1.fillna(0)
    timeseries_diff2 = timeseries_diff2.fillna(0)

    timeseries_adf = ADF(timeseries['data'].tolist())
    timeseries_diff1_adf = ADF(timeseries_diff1['data'].tolist())
    timeseries_diff2_adf = ADF(timeseries_diff2['data'].tolist())

    print('timeseries_adf : ', timeseries_adf)
    print('timeseries_diff1_adf : ', timeseries_diff1_adf)
    print('timeseries_diff2_adf : ', timeseries_diff2_adf)

    plt.figure(figsize=(12, 8))
    # plt.plot(timeseries, label='Original', color='blue')
    plt.plot(timeseries_diff1, label='Diff1', color='red')
    plt.plot(timeseries_diff2, label='Diff2', color='purple')
    plt.legend(loc='best')
    plt.show()


def autocorrelation(timeseries, lags):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(timeseries, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(timeseries, lags=lags, ax=ax2)
    plt.show()


#1导入数据
seq_train, seq_test = generate_purchase_seq()
#2差分数据
diff(seq_train)
#3得出1阶差分满足要求，从而进行1阶差分
seq_train_diff = seq_train.diff(1)
seq_train_diff = seq_train_diff.fillna(0)
#4
autocorrelation(seq_train_diff, 20)

trend_evaluate = sm.tsa.arma_order_select_ic(seq_train_diff, ic=['aic'], trend='nc', max_ar=4, max_ma=4)
print('calculate AIC', trend_evaluate.aic_min_order)
# print('trend BIC', trend_evaluate.bic_min_order)

# 序列模型训练
def ARIMA_Model(timeseries, order):
    model = ARIMA(timeseries, order=order)
    return model.fit(disp=0)


model = ARIMA_Model(seq_train, (2, 1, 4))
pred_day = 8
y_forecasted =model.forecast(steps=pred_day, alpha=0.01)[0] #作为期8天的预测
y_truth = seq_test['2018-12-24':'2018-12-31'].values

plt.figure(2)
plt.plot(y_forecasted, color='red', label='predict_seq')
plt.plot(y_truth, color='blue', label='seq_test')
plt.legend()
plt.show()

