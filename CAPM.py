import pandas as pd
import seaborn as sns
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go


stocks_df = pd.read_csv('C:/Users/ECEM YAMAN/Desktop/MATLAB/FİNANSAL ANALİZ/stock.csv')
stocks_df
stocks_df = stocks_df.sort_values(by = ['Date'])
stocks_df
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
  interactive_plot(stocks_df, 'Prices')
  
def daily_return(df):

  df_daily_return = df.copy()
  
  # Her hisse senedinin gözden geçirilmesi
  for i in df.columns[1:]:
    
    # Stoğa ait her satır için döngü oluşturulması
    for j in range(1, len(df)):
      
      # Önceki güne göre değişim yüzdesi hesaplama
      df_daily_return[i][j] = ((df[i][j]- df[i][j-1])/df[i][j-1]) * 100
    
    # önceki değer mevcut olmadığından ilk satırın değerini sıfır olarak ayarlama
    df_daily_return[i][0] = 0
  return df_daily_return

stocks_daily_return = daily_return(stocks_df)
stocks_daily_return
stocks_daily_return['AAPL']
stocks_daily_return['sp500']
stocks_daily_return.plot(kind = 'scatter', x = 'sp500', y = 'AAPL')
beta, alpha = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return['AAPL'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('AAPL', beta, alpha)) 
stocks_daily_return.plot(kind = 'scatter', x = 'sp500', y = 'AAPL')
plt.plot(stocks_daily_return['sp500'], beta * stocks_daily_return['sp500'] + alpha, '-', color = 'r')

beta, alpha = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return['TSLA'], 1)
print(beta)  
# Şimdi dağılım grafiğini ve üzerine düz çizgiyi ekleme
stocks_daily_return.plot(kind = 'scatter', x = 'sp500', y = 'TSLA')

# Alfa ve beta parametreli düz çizgi denklemi
# Düz çizgi denklemi y = beta * rm + alpha
plt.plot(stocks_daily_return['sp500'], beta * stocks_daily_return['sp500'] + alpha, '-', color = 'r')

beta
stocks_daily_return['sp500'].mean()

rm = stocks_daily_return['sp500'].mean() * 252
rm

rf = 0 
ER_AAPL = rf + ( beta * (rm-rf) ) 

# Önce Amazon için Beta'yı hesaplamamız gerekir.
beta, alpha = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return['T'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('T', beta, alpha)) 
ER_T = rf + ( beta * (rm - rf) ) 
print(ER_T)


beta = {}
alpha = {}

# Her hisse senedinin günlük getirisi
for i in stocks_daily_return.columns:

  # Tarih ve S&P 500 Sütunlarını yok saymak 
  if i != 'Date' and i != 'sp500':
   # her bir hisse senedi ile Sp500 (Pazar) arasında bir dağılım grafiği çizme
    stocks_daily_return.plot(kind = 'scatter', x = 'sp500', y = i)
    
    # Her stok ile SP500 arasına bir polinom yerleştirme (Sıralı poli = 1 düz bir çizgidir)
    b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[i], 1)
    
    plt.plot(stocks_daily_return['sp500'], b * stocks_daily_return['sp500'] + a, '-', color = 'r')
    
    beta[i] = b
    
    alpha[i] = a
    
    plt.show()


for i in stocks_daily_return.columns:
  
  if i != 'Date' and i != 'sp500':
    
    # Sp500'e karşı her hisse senedi dağılım grafiğini çizmek için grafiksel ifade kullanma
    fig = px.scatter(stocks_daily_return, x = 'sp500', y = i, title = i)

    # Verilere düz bir çizgi ekleyerek beta ve alfa elde etme
    b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[i], 1)
    
    # Düz çizgiyi çizme
    fig.add_scatter(x = stocks_daily_return['sp500'], y = b*stocks_daily_return['sp500'] + a)
    fig.show()
    
    
beta
alpha
keys = list(beta.keys())
keys
ER = {}

rf = 0 # risksiz oranın sıfır olduğunu varsayıyoruz.
rm = stocks_daily_return['sp500'].mean() * 252 # bu piyasanın beklenen getirisidir
rm

for i in keys:
  # CAPM kullanarak her güvenlik için getiriyi hesaplama
  ER[i] = rf + ( beta[i] * (rm-rf) ) 
  
for i in keys:
  print('CAPM ye Göre Beklenen Getiri: {}%'.format(i, ER[i]))

portfolio_weights = 1/8 * np.ones(8) 
portfolio_weights
ER_portfolio = sum(list(ER.values()) * portfolio_weights)
ER_portfolio
print('Portföy için CAPM Bazında Beklenen Getiri {}%\n'.format(ER_portfolio))

ER['AMZN']


