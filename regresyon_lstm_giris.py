import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import plotly.figure_factory as ff
from plotly.figure_factory import create_distplot
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow import keras

#*********************** VERİ KİTAPLARI VE KÜTÜPHANELERi YÜKLEME ***********************************

stock1 = pd.read_csv('C:/Users/ECEM YAMAN/Desktop/FİNANSAL ANALİZ/stock.csv')
stock1
stock2 = pd.read_csv('C:/Users/ECEM YAMAN/Desktop/FİNANSAL ANALİZ/stock_volume.csv')
stock2

stock1 = stock1.sort_values(by = ['Date'])
stock1
#stock1.info()
stock1.isnull().sum()
stock1 = stock1.sort_values(by = ['Date'])
stock2
stock2.isnull().sum()
#stock2.info()
stock2.describe()


#*********************** VERİ GÖRSELLEŞTİRME ***********************************

def normalize(stock1):
  x = stock1.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x

def interactive_plot(stock1, title):
  fig = px.line(title = title)
  for i in stock1.columns[1:]:
    fig.add_scatter(x = stock1['Date'], y = stock1[i], name = i) 
  fig.show()
interactive_plot(stock1, 'FİYATLAR')

# Tarihi, hisse senedi fiyatını ve hacmi birleştirme
def hisse_senedi(fiyat, hacim, name):
    return pd.DataFrame({'Date': fiyat['Date'], 'Close': fiyat[name], 'Volume': hacim[name]})

def pencere(data):
   #1 günlük pencere
  n = 1
  # 1 gün için fiyatları içeren sütunlar
  data['Target'] = data[['Close']].shift(-n)
  # yeni oluşan veri setini dödürdük
  return data

# AAPL için hisse senedi fiyatlarını ve hacimleri aldık.
sonuc = hisse_senedi(stock1, stock2, 'AAPL')
sonuc

sonuc2 = pencere(sonuc)
sonuc

sonuc2 = sonuc2[:-1]
sonuc2

# VERİ ÖLÇEKLENDİRME
olcekleme = MinMaxScaler(feature_range = (0, 1))
sonuc2 = olcekleme.fit_transform(sonuc2.drop(columns = ['Date']))
sonuc2.shape

# TRAINIG DATA VE TEST DATA OLUSTURMA

X = sonuc2[:,:2]
y = sonuc2[:,2:]
X.shape, y.shape


split = int(0.65 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]
X_train.shape, y_train.shape
X_test.shape, y_test.shape

def show_plot(data, title):
  plt.figure(figsize = (13, 5))
  plt.plot(data, linewidth = 3)
  plt.title(title)
  plt.grid()

# verileri  plot ile ekranda grafik halinde gösterdik.

show_plot(X_train, 'Training Data')
show_plot(X_test, 'Testing Data')


#************************RIDGE DOĞRUSAL REGRESYON MODELİ **************************

regression_model = Ridge()
regression_model.fit(X_train, y_train)

#Modeli tahmin etme ve dogrulugunu hesaplama 
dogruluk = regression_model.score(X_test, y_test)
print("Linear Regression Sonucu: ", dogruluk)
tahmini_fiyatlar = regression_model.predict(X)
tahmini_fiyatlar

# tahmin edilen değerler bir listeye aktarılır.
tahmin_edilen = []
for i in tahmini_fiyatlar:
  tahmin_edilen.append(i[0])
len(tahmin_edilen)
close = []
for i in sonuc2:
  close.append(i[0])
  
# Ayrı stok verilerindeki tarihlere göre dataframe olusturma
tahmin = sonuc2[['Date']]
tahmin
# dataframe' e yakın degerleri ekledik
tahmin['Close'] = close
tahmin
#Tahmin edilen değerleri dataframe'e ekleyin
tahmin['Prediction'] = tahmin
tahmin
interactive_plot(tahmin, "ORJINAL & TAHMIN")


#####################################################################

sp500 = hisse_senedi(stock1, stock2, 'sp500')
sp500
training_data = sp500.iloc[:, 1:3].values
training_data
#┬ verilere normalizasyon islemi yaptık
olcekleme = MinMaxScaler(feature_range = (0, 1))
training_set_olceklendi = olcekleme.fit_transform(training_data)
#
X = []
y = []
for i in range(1, len(sonuc2)):
    X.append(training_set_olceklendi [i-1:i, 0])
    y.append(training_set_olceklendi [i, 0])
#Veriler dizi formatına dönüştürüldü
X = np.asarray(X)
y = np.asarray(y)

split = int(0.7 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

## Modeldeki 1D dizilerini 3B dizilere çevrildi
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train.shape, X_test.shape

# model oluşturma
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences= True)(inputs)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150)(x)
outputs = keras.layers.Dense(1, activation='lineer')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss="mse")
model.summary()

history = model.fit(
    X_train, y_train,
    epochs = 20,
    batch_size = 32,
    validation_split = 0.2
)
tahmin = model.predict(X)
tahmin1 = []

for i in tahmin:
  tahmin1.append(i[0][0])
  tahmin1






























