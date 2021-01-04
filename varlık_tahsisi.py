import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

stock1 = pd.read_csv('C:/Users/ECEM YAMAN/Desktop/MATLAB/FİNANSAL ANALİZ/stock.csv')
stock1
# Verileri Tarihe göre sıralama
stock1 = stock1.sort_values(by = ['Date'])
stock1

# Ham stok verilerini ve normalleştirilmiş olanları görselleştirmek için Plotly kullanımı
def normalize(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x

def interactive_plot(df, title):
  fig = px.line(title = title)
  for i in df.columns[1:]:
    fig.add_scatter(x = df['Date'], y = df[i], name = i)
  fig.show()
interactive_plot(stock1, 'HİSSE SENETLERİ')
interactive_plot(normalize(stock1), 'NORMALİZE UYGULANMIŞ HİSSE SENETLERİ')

# RASTGELE VARLIK TAHSİSİ YAPMA VE GÜNLÜK PORTFÖY HESAPLAMA 

# Portföy ağırlıklarının toplamı 1 olmalıdır
# Rastgele portföy ağırlıkları oluşturma
np.random.seed()

# # Stoklar için rastgele ağırlıklar oluşturma ve bunları normalleştirme
weights = np.array(np.random.random(9))

# Ağırlıklar toplamı 1 olmalı
weights = weights / np.sum(weights) 
print(weights)

# Stok değerlerini normalleştirin
portfoy = normalize(stock1)
portfoy

portfoy.columns[1:]
# sayacı döndürür.
for counter, stock in enumerate(portfoy.columns[1:]):
  portfoy[stock] = portfoy[stock] * weights[counter]
  portfoy[stock] = portfoy[stock] * 1000000
portfoy
portfoy['GUNLUK PORTFOY DEGERİ ($) CİNSİNDEN '] = portfoy[portfoy != 'Date'].sum(axis = 1)
portfoy
portfoy['portföy günlük getiri yüzdesi'] = 0.0000

for i in range(1, len(stock1)):
 # Önceki güne göre değişim yüzdesini hesaplama
  portfoy['portföy günlük getiri yüzdesi'][i] = ( (portfoy['GUNLUK PORTFOY DEGERİ ($) CİNSİNDEN '][i] - portfoy['GUNLUK PORTFOY DEGERİ ($) CİNSİNDEN '][i-1]) / portfoy['GUNLUK PORTFOY DEGERİ ($) CİNSİNDEN '][i-1]) * 100
portfoy
# # Yatırım yapılacak 1.000.000 $ 'a sahip olduğumuzu varsayalım ve bu fonu hisse senetlerinin ağırlıklarına göre tahsis edeceğiz
# Ağırlıkların yanı sıra hisse senedi fiyatlarını da alan ve geri dönen bir fonksiyon oluşturacağız:
# (1) Belirtilen süre boyunca her bir menkul kıymetin $ cinsinden günlük değeri
# (2) Tüm portföyün genel günlük değeri
# (3) Günlük getiri

def portfoy_tahsisi(df, weights):
  portfoy = df.copy()
 # Hisse senedi değerlerini normalleştirme
  portfoy = normalize(portfoy)
for counter, stock in enumerate(portfoy.columns[1:]):
    portfoy[stock] = portfoy[stock] * weights[counter]
    portfoy[stock] = portfoy[stock] * 1000000
portfoy['GUNLUK PORTFOY DEGERİ ($) CİNSİNDEN '] = portfoy[portfoy != 'Date'].sum(axis = 1)
portfoy['portföy günlük getiri yüzdesi'] = 0.0000

for i in range(1, len(stock1)):
    # # Önceki güne göre değişim yüzdesini hesaplama
    portfoy['portföy günlük getiri yüzdesi'][i] = ( (portfoy['GUNLUK PORTFOY DEGERİ ($) CİNSİNDEN'][i] - portfoy['GUNLUK PORTFOY DEGERİ ($) CİNSİNDEN'][i-1]) / portfoy['GUNLUK PORTFOY DEGERİ ($) CİNSİNDEN'][i-1]) * 100 
  # önceki değer mevcut olmadığından ilk satırın değerini sıfır olarak ayarladık.
portfoy['portföy günlük getiri yüzdesi'][0] = 0
portfoy

# Portföyün genel günlük değer ve zaman grafiğini çizimi
fig = px.line(x = portfoy.Date, y = portfoy['portföy günlük getiri yüzdesi'], title = 'portföy günlük getiri yüzdesi')
fig.show()
# Normalizasyon uygulanan tüm hisse senetlerinin grafiğini çizme
interactive_plot(portfoy.drop(['GUNLUK PORTFOY DEGERİ ($) CİNSİNDEN ', 'portföy günlük getiri yüzdesi'], axis = 1), 'Portföy bireysel hisse senetleri, $ değerinde zaman içinde değişimi')
# Günlük getirilerin bir histogramını çizdirme
fig = px.histogram(portfoy, x = 'portföy günlük getiri yüzdesi')
fig.show()

fig = px.line(x = portfoy.Date, y = portfoy['GUNLUK PORTFOY DEGERİ ($) CİNSİNDEN'], title= '($) Bazında Portföy Toplam Değeri')
fig.show()
# Portföyün kümülatif getirisi (Şimdi portföyün başlangıç ​​değerine kıyasla son net değerini arıyoruz.)
cummulative_return = ((portfoy['GUNLUK PORTFOY DEGERİ ($) CİNSİNDEN '][-1:] - portfoy['GUNLUK PORTFOY DEGERİ ($) CİNSİNDEN '][0])/ portfoy['GUNLUK PORTFOY DEGERİ ($) CİNSİNDEN '][0]) * 100
print('Portföyün kümülatif getirisi {} %'.format(cummulative_return.values[0]))
# Portföy standart sapmasını hesaplama
print('Portföyün standart sapması {}'.format(portfoy['portföy günlük getiri yüzdesi'].std()))
# Günlük ortalama getiriyi hesaplama
print('Portföyün ortalama günlük getirisi {} %'.format(portfoy['portföy günlük getiri yüzdesi'].mean()))
sharpe_orani = portfoy['portföy günlük getiri yüzdesi'].mean() / portfoy['portföy günlük getiri yüzdesi'].std() * np.sqrt(252)
print('Portföyün Sharpe oranı {}'.format(sharpe_orani))

