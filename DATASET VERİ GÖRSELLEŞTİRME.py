import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
import plotly.figure_factory as ff
from plotly.figure_factory import create_distplot


stock1 = pd.read_csv('C:/Users/ECEM YAMAN/Desktop/FİNANSAL ANALİZ/stock.csv')  # dosyamızı okuduk
stock2 = pd.read_csv('C:/Users/ECEM YAMAN/Desktop/FİNANSAL ANALİZ/stock_volume.csv')

stock1.mean()
stock1.std()
stock1.isnull().sum()

#stock1.info() # Datasetimiz içerisindeki veri tipleriyle ile ilgili bilgi aldık.
#stock2.info()

def show_plot(stock1, fig_title):   #Plotly kullanarak etkileşimli grafikleri çizme
  stock1.plot(x = 'Date', figsize = (15,7), linewidth = 3, title = fig_title)
  plt.grid()
  plt.show()
show_plot(stock1, 'HAM STOK FİYATLARI (NORMALİZASYON OLMADAN)')

#Plotly express kullanarak etkileşimli bir veri çizimi gerçekleştirme 

"""def interactive_plot(stock1, title):
  fig = px.line(title = title)
  for i in stock1.columns[1:]:
    fig.add_scatter(x = stock1['Date'], y = stock1[i], name = i) 
  fig.show()
interactive_plot(stock1, 'PRICES')

# BİREYSEL HİSSE SENETLERİNİ HESAPLAMA
sp = stock1['sp500']
sp2 = sp.copy()

#Veri setindeki her öğeyi döngüye aldık.
for j in range(1, len(sp)):

  # Önceki güne göre değişim yüzdesini hesaplarız
  sp2[j] = ((sp[j]- sp[j-1])/sp[j-1]) * 100

# ilk satır öğesine sıfır koyarız
sp2[0] = 0
sp2   
"""

# BİRDEN FAZLA HİSSE SENEDİ GÜNLÜK İADE HESAPLAMA

 # Hisse senetlerinin günlük getirilerini hesaplamak için bir fonksiyon tanımladık
def g_getiri(stock1):
  gunluk_getiri = stock1.copy()
  for i in stock1.columns[1:]:
    
    # Stoğa ait her satır için döngü yazıldı.
    for j in range(1, len(stock1)):

      #Önceki güne göre değişim yüzdesini hesaplandı
      gunluk_getiri[i][j] = ((stock1[i][j]- stock1[i][j-1])/stock1[i][j-1]) * 100
    
    # ilk satırın değerini sıfır olarak ayarlandı.
    gunluk_getiri[i][0] = 0
  
  return gunluk_getiri


stok_gunluk_getiri = g_getiri(stock1)
stok_gunluk_getiri

# GÜNLÜK İADELER ARASINDAKİ İLİŞKİLERİ HESAPLA

# Günlük Getiri Korelasyonu
cm = stok_gunluk_getiri.drop(columns = ['Date']).corr()
plt.figure(figsize=(10, 10))
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax);

stok_gunluk_getiri.hist(figsize=(10, 10), bins = 40); 

################################### TEKRAR KONTROL ET ###################################

# Tüm datasetleri bir listede gruplama
# Günlük iade verilarin bir kopyasını oluşturma
stock_histogram = stok_gunluk_getiri.copy()
stock_histogram = stock_histogram.drop(columns = ['Date'])
data = []
# Sütunlar döngüye alınır.
for i in stock_histogram.columns:
  data.append(stok_gunluk_getiri[i].values)
data
fig = ff.create_distplot(data, stock_histogram.columns)
fig.show()

######################################################################
""""
def normalize(stock1):
  x = stock1.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x
show_plot(normalize(stock1), 'NORMALLEŞTİRİLMİŞ STOK FİYATLARI')
show_plot(stok_gunluk_getiri, 'GÜNLÜK STOK GETİRİLERİ')


"""

