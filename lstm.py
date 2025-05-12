# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:34:52 2025

@author: koray
"""

#%% loadd dataset and preprocessing
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder # Etiketleri sayısal formata çevirir
from sklearn.model_selection import train_test_split


from tensorflow.keras.preprocessing.text import Tokenizer # metin verisini sayılara çevirir
from tensorflow.keras.preprocessing.sequence import pad_sequences # dizileri aynı uzunluga getirir
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2

from config import parse_arguments
import os
import pickle

#%% take arguments

args=parse_arguments()
epochs=args.epochs
patience=args.patience
batch_size=args.batch_size
num_words=args.num_words
maxlen=args.maxlen
output_dim=args.output_dim
lstm_units=args.lstm_units

#%%
newsgroup=fetch_20newsgroups(subset="all") # all ile eğitim ve test verileri yüklendi
X=newsgroup.data
y=newsgroup.target

#%%
#Metinleri tokenları ayırma.Cümle içindeki bütün kelimeleri ayırma ve padding uygulama
tokenizer=Tokenizer(num_words=num_words)#10000 tane kelime kullanılsın en çok kullanılan
tokenizer.fit_on_texts(X)#tokenizer'i metin verisi ile fit edelim
X_sequences=tokenizer.texts_to_sequences(X)
X_padded=pad_sequences(X_sequences,maxlen=maxlen)# Metinleri ayın uzunluğa getirir

#%%

#Etiketleri sayısal hale donustur
label_encoder=LabelEncoder()
y_encoded=label_encoder.fit_transform(y)

#%%

#train test split yapıldı.
X_train,X_test,y_train,y_test=train_test_split(X_padded,y_encoded,test_size=0.2,random_state=42)

#%% create lstm model
from tensorflow.keras import backend as K# tensorflow un matematikse işlemleri için kullanılan kütüphane

def f1_score(y_true,y_pred):

    y_pred = K.argmax(y_pred, axis=1)
    y_pred=K.round(y_pred) #Tahminler 0 veya 1 e yuvarlanır

   
    tp=K.sum(K.cast(y_true*y_pred,"float"),axis=0)#true positive
    fp=K.sum(K.cast((1-y_true)*y_pred,"float"),axis=0)#false positive
    fn=K.sum(K.cast(y_true*(1-y_pred),"float"),axis=0)#false negative
    
    precision=tp/(tp+fp+K.epsilon())
    recall=tp/(tp+fn+K.epsilon())
    
    f1=2*(precision*recall)/(precision +recall +K.epsilon())
    
    return K.mean(f1)

def build_lstm_model():
    
    model=Sequential()
    
    #input_dim=kelime vektörlerinin toplam boyutu.10000 kelime var
    #output_dim=kelime vektörlerinin boyutu. Bri kelime 64 uznlugnda
    #input_length= her metinin uzunluğu
    model.add(Embedding(input_dim=10000,output_dim=output_dim,input_length=maxlen))

    #return_sequence sonucların tum zaman adaımları yerine sdece son adımda return etmesi
    #model.add(LSTM(units=lstm_units,return_sequences=False))#64 adet hücre
    model.add(LSTM(units=lstm_units, return_sequences=False, dropout=0.3, recurrent_dropout=0.3))
    #dropout
    model.add(Dropout(0.5))
    
    #Dense
    model.add(Dense(20, activation="softmax", kernel_regularizer=l2(0.001)))
    
    #model compile
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy",f1_score])
    return model


#%%model oluşturma
model=build_lstm_model()
model.summary()

#%% training


early_stopping=EarlyStopping(monitor="val_loss",patience=patience,restore_best_weights=True)

history=model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_split=0.3,callbacks=early_stopping,verbose=2)
epochs=len(history.history["val_accuracy"])+1
# %%save

loss = float(history.history['loss'][-1])
val_loss = float(history.history['val_loss'][-1])
acc = float(history.history['accuracy'][-1])
val_acc = float(history.history['val_accuracy'][-1])

folder_name = "lstm_model"
os.makedirs(folder_name, exist_ok=True)

'''
file_name = f"epoch{epochs}_loss{loss:.2f}_val_loss{val_loss:.2f}_acc{acc:2f}_val_acc{val_acc:.2f}.h5"
save_path = os.path.join(folder_name, file_name)
model.save(save_path)

'''
existing_files = os.listdir(folder_name)
model_nums = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.startswith("model_") and f.endswith(".h5")]

next_num = max(model_nums) + 1 if model_nums else 1
model_filename = f"model_{next_num:02d}.h5"
save_path = os.path.join(folder_name, model_filename)
model.save(save_path)


log_path = os.path.join(folder_name, "log.txt")
with open(log_path, "a") as f:
    f.write(f"{model_filename}: epoch={epochs}, loss={loss:.4f}, val_loss={val_loss:.4f}, acc={acc:.4f}, val_acc={val_acc:.4f}\n")

print(f"Model kaydedildi: {save_path}")
#%% save tokenizer and labelencoder for testing
save_dir = os.path.join("exports", model_filename)
os.makedirs(save_dir, exist_ok=True)


tokenizer_path = os.path.join(save_dir, 'tokenizer.pkl')
label_encoder_path = os.path.join(save_dir, 'label_encoder.pkl')

# Eski dosyaları sil (varsa)
for path in [tokenizer_path, label_encoder_path]:
    if os.path.exists(path):
        os.remove(path)
        print(f"Silindi: {path}")

with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

# %% model evaluation

loss,accuracy,f1_score=model.evaluate(X_test,y_test,verbose=2)
print(f"Test loss: {loss: .4f}, Test accuracy: {accuracy: .4f}, F1_Score: {f1_score}")

plt.figure()

#traingg loss ve val loss
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="Training Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")
plt.title("Traning and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid("True")


# training accuracy ve val accuracy
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")
plt.title("Traning and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid("True")


save_dir=os.path.join("Plots",model_filename)
os.makedirs(save_dir,exist_ok=True)

plot_path = os.path.join(save_dir, "loss_accuracy_plot.png")
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print(f"Grafik kaydedildi: {plot_path}")
