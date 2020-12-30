import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from keras.optimizers import SGD

creditcard = pd.read_csv('C:/Users/ECEM YAMAN/Desktop/MATLAB/FİNANSAL ANALİZ/marketing_data.csv')
creditcard
#yazılan fiyata göre müsteri cekme
creditcard[creditcard['ONEOFF_PURCHASES'] == 40761.25]
creditcard['CASH_ADVANCE'].max()
creditcard[creditcard['CASH_ADVANCE'] == 47137.211760000006]

# VERİ KÜMESİNİ GÖRSELLEŞTİRME

creditcard.isnull().sum()
#Eksik öğeler 'MINIMUM_PAYMENT' ortalamasıyla doldurulur
creditcard.loc[(creditcard['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = creditcard['MINIMUM_PAYMENTS'].mean()
#Eksik öğeler 'CREDIT_LIMIT' ile doldurulur
creditcard.loc[(creditcard['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = creditcard['CREDIT_LIMIT'].mean()
sns.heatmap(creditcard.isnull(), yticklabels = False, cbar = False, cmap="Blues")
creditcard.duplicated().sum()
creditcard.drop("CUST_ID", axis = 1, inplace= True)
creditcard.head()
n = len(creditcard.columns)
n
creditcard.columns

# distplot, matplotlib.hist işlevini seaborn kdeplot() ile birleştirir
# KDE Grafiği Kernel Yoğunluk Tahminini temsil eder
# KDE, sürekli bir değişkenin Olasılık Yoğunluğunu görselleştirmek için kullanılır.
# KDE, sürekli bir değişkende farklı değerlerde olasılık yoğunluğunu gösterir. 

plt.figure(figsize=(10,50))
for i in range(len(creditcard.columns)):
  plt.subplot(17, 1, i+1)
  sns.distplot(creditcard[creditcard.columns[i]], kde_kws={"color": "b", "lw": 3, "label": "KDE"}, hist_kws={"color": "g"})
  plt.title(creditcard.columns[i])

## "PURCHASES", tek seferlik alımlar, "taksitli alımlar, satın alma işlemleri, kredi limiti ve ödemeler arasında yüksek korelasyona sahiptir.
# 'PURCHASES_FREQUENCY' ve 'PURCHASES_INSTALLMENT_FREQUENCY' arasında Pozitif ilişki vardır.

plt.tight_layout()
korelasyon = creditcard.corr()
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(korelasyon, annot = True)

# ELBOW METHODUNU KULLANARAK EN İYİ KÜMELEME SAYISINI BULMA

# ELBOW METODU, bir veri kümesindeki uygun sayıda kümenin bulunmasına yardımcı olmak için tasarlanmış 
#küme analizi içinde tutarlılığın yorumlanması ve doğrulanması için sezgisel bir yöntemdir.

scaler = StandardScaler()
creditcard_scaled = scaler.fit_transform(creditcard)
creditcard_scaled.shape
creditcard_scaled

scores_1 = []
range_values = range(1, 8)

for i in range_values:
  kmeans = KMeans(n_clusters = i)
  kmeans.fit(creditcard_scaled)
  scores_1.append(kmeans.inertia_) 

plt.plot(scores_1, 'bx-')
plt.title('DOGRU SAYIDA KÜMELERİ BULALIM....')
plt.xlabel('KUMELER')
plt.ylabel('PUANLAR') 
plt.show()

# K-MEANS metodunu kullanma

kmeans = KMeans(8)
kmeans.fit(creditcard_scaled)
labels = kmeans.labels_
kmeans.cluster_centers_.shape
cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [creditcard.columns])
cluster_centers    
# sayıların ne anlama geldiğini görmek için ters dönüşüm yaptık.
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [creditcard.columns])
cluster_centers

labels.shape # veri noktalarıyla ilişkili etiketler
labels.max()
labels.min()
y_kmeans = kmeans.fit_predict(creditcard_scaled)
y_kmeans


#küme etiketlerini orijinal data frameile birleştirme
creditcard_cluster = pd.concat([creditcard, pd.DataFrame({'cluster':labels})], axis = 1)
creditcard_cluster.head()

# Çeşitli kümelerin histogramı 

for i in creditcard.columns:
  plt.figure(figsize = (35, 5))
  for j in range(8):
    plt.subplot(1,8,j+1)
    cluster = creditcard_cluster[creditcard_cluster['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{}    \nCluster {} '.format(i,j))
  plt.show()


#TEMEL BİLEŞEN ANALİZİNİ UYGULAMA VE SONUÇLARI GÖRSELLEŞTİRME

# Ana bileşenleri edindik
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(creditcard_scaled)
principal_comp
# İki bileşenle bir data frame oluşturma
pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])
pca_df.head()
#Küme etiketlerini dataframeler ile birleştirme
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
pca_df.head()
plt.figure(figsize=(10,10))
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','pink','yellow','gray','purple', 'black'])
plt.show()

# AUTOENCODERS UYGULAMA

encoding_dim = 7
input_df = Input(shape=(17,))

# Glorot normal başlatıcı (Xavier normal başlatıcı) kesilmiş bir normal dağılımdan örnekler alır

x = Dense(encoding_dim, activation='relu')(input_df)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(2000, activation='relu', kernel_initializer = 'glorot_uniform')(x)
encoded = Dense(10, activation='relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(2000, activation='relu', kernel_initializer = 'glorot_uniform')(encoded)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)
decoded = Dense(17, kernel_initializer = 'glorot_uniform')(x)

# autoencoder
autoencoder = Model(input_df, decoded)

# autoencoder - boyut küçültülmesi için kullanılır
encoder = Model(input_df, encoded)
autoencoder.compile(optimizer= 'adam', loss='mean_squared_error')
creditcard_scaled.shape
autoencoder.fit(creditcard_scaled, creditcard_scaled, batch_size = 128, epochs = 25,  verbose = 1)
autoencoder.save_weights('autoencoder.h5')
pred = encoder.predict(creditcard_scaled)
pred.shape
scores_2 = []
range_values = range(1, 20)

for i in range_values:
  kmeans = KMeans(n_clusters= i)
  kmeans.fit(pred)
  scores_2.append(kmeans.inertia_)

plt.plot(scores_2, 'bx-')
plt.title('Finding right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('scores') 
plt.show()
plt.plot(scores_1, 'bx-', color = 'r')
plt.plot(scores_2, 'bx-', color = 'g')
kmeans = KMeans(4)
kmeans.fit(pred)
labels = kmeans.labels_
y_kmeans = kmeans.fit_predict(creditcard_scaled)
df_cluster_dr = pd.concat([creditcard, pd.DataFrame({'cluster':labels})], axis = 1)
df_cluster_dr.head()
pca = PCA(n_components=2)
prin_comp = pca.fit_transform(pred)
pca_df = pd.DataFrame(data = prin_comp, columns =['pca1','pca2'])
pca_df.head()
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
pca_df.head()
plt.figure(figsize=(10,10))
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','yellow'])
plt.show()

