import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import plotly.express as px

# Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

stock_df = pd.read_csv('C:/Users/ECEM YAMAN/Desktop/MATLAB/FİNANSAL ANALİZ/stock_sentiment.csv')
stock_df

#Metinden zararlı kelimeleri kaldırma

import string
string.punctuation

Test = '$AI & Machine learning'
Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join

Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed

# Kelime dizisini  oluşturmak için karakterleri yeniden birleştirme

Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join

#Noktalama işaretlerini kaldırmak için fonksiyon oluşturuldu.
def remove_punc(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)

    return Test_punc_removed_join

# VERİ SETİNDEN NOKTALAMA İŞARETLERİNİ KALDIRMA 
stock_df['Text Without Punctuation'] = stock_df['Text'].apply(remove_punc)
stock_df
stock_df['Text'][2]
stock_df['Text Without Punctuation'][2]

#ENGELLENECEK KELİMELER
nltk.download("stopwords")
stopwords.words('english')

#NLTK' DAN FARKLI OLARAK ENGELLENECEK KELİMELER
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','will','aap','co','day','user','stock','today','week','year'])

# ENGELLENECEK KELİMELERİ VE 2 KELİMEDEN AZ KISA KELİMELERİ KALDIRMA
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token) >= 3 and token not in stop_words:
            result.append(token)
            
    return result

# METİN SÜTUNUNA ÖN İŞLEME UYGULAMA

stock_df['Text Without Punc & Stopwords'] = stock_df['Text Without Punctuation'].apply(preprocess)
stock_df['Text'][0]
stock_df['Text Without Punc & Stopwords'][0]
stock_df

# KELİMELERİ BİR DİZİDE BİRLEŞTİRME
stock_df['Text Without Punc & Stopwords Joined'] = stock_df['Text Without Punc & Stopwords'].apply(lambda x: " ".join(x))

# poziTİF duygu analizi içeren metinlerin wordcloud'unu plot etme
plt.figure(figsize = (20, 20)) 
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800).generate(" ".join(stock_df[stock_df['Sentiment'] == 1]['Text Without Punc & Stopwords Joined']))
plt.imshow(wc, interpolation = 'bilinear');

# engellenmiş kelimeler sonrası veri kümesi görselleştirme

stock_df
nltk.download('punkt')
# word_tokenize: bir dizeyi kelimelere ayırmak için kullanılır.

print(stock_df['Text Without Punc & Stopwords Joined'][0])
print(nltk.word_tokenize(stock_df['Text Without Punc & Stopwords Joined'][0]))

#Belgedeki maksimum veri uzunluğu elde edilir.
maxlen = -1
for doc in stock_df['Text Without Punc & Stopwords Joined']:
    tokens = nltk.word_tokenize(doc)
    if(maxlen < len(tokens)):
        maxlen = len(tokens)
print("The maximum number of words in any document is:", maxlen)

tweets_length = [ len(nltk.word_tokenize(x)) for x in stock_df['Text Without Punc & Stopwords Joined'] ]
tweets_length
# bir metindeki kelime sayının dağılımını çizme
fig = px.histogram(x = tweets_length, nbins = 50)
fig.show()

stock_df
# bir veri setinde bulunan toplam kelimeler elde edilir. 

list_of_words = []
for i in stock_df['Text Without Punc & Stopwords']:
    for j in i:
        list_of_words.append(j)
list_of_words
#benzersiz kelimelerin toplam sayısı elde edilir.
total_words = len(list(set(list_of_words)))
total_words
# veriler test verisi ve eğitim verisi olarak ayrılır.
X = stock_df['Text Without Punc & Stopwords']
y = stock_df['Sentiment']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
X_train.shape
X_test.shape
X_train
# kelimeleri belirtmek ve belirtilen kelime dizilerinn oluşturılması için gerekn işlem sağlanır.
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(X_train)
# eğitim verileri
train_sequences = tokenizer.texts_to_sequences(X_train)
#test verileri
test_sequences = tokenizer.texts_to_sequences(X_test)
train_sequences
test_sequences
print("The encoding for document\n", X_train[1:2],"\n is: ", train_sequences[1])
# eğitim ve test verilerine dolgu ekleme
padded_train = pad_sequences(train_sequences, maxlen = 29, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences, maxlen = 29, truncating = 'post')

for i, doc in enumerate(padded_train[:3]):
     print("The padded encoding for document:", i+1," is:", doc)
     
# 2D veri dönüştürme
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)
y_test_cat.shape
y_train_cat.shape

#Eğitim verisine ve test verisine dolgu ekleme yapılır.
padded_train = pad_sequences(train_sequences, maxlen = 15, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences, maxlen = 15, truncating = 'post')

########### RNN VE LSTM ##############

#sıralı bir model oluşturma
model = Sequential()
#embed layer
model.add(Embedding(total_words, output_dim = 512))
#çift yönlü RNN VE LSTM
model.add(LSTM(256))
# Dense Layers
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
model.summary()
#Modeli eğitme
model.fit(padded_train, y_train_cat, batch_size = 32, validation_split = 0.2, epochs = 10)


#tahmin etme işlemi
pred = model.predict(padded_test)
prediction = []
for i in pred:
  prediction.append(np.argmax(i))

#orjinal değerleri içeren liste  
original = []
for i in y_test_cat:
  original.append(np.argmax(i))

# metin verilerinde doğruluk puanı
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(original, prediction)
accuracy
#confusion matrisini çizme 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(original, prediction)
sns.heatmap(cm, annot = True)
