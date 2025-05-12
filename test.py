# -*- coding: utf-8 -*-
"""
Created on Sun May  4 15:51:10 2025

@author: koray
"""
from config import parse_arguments
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import os
from sklearn.datasets import fetch_20newsgroups

args = parse_arguments()
maxlen = args.maxlen

newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

def load_tokenizer_and_label_encoder(exports_dir):
    # exports klasöründeki yalnızca klasörleri al
    model_dirs = [
        d for d in os.listdir(exports_dir)
        if os.path.isdir(os.path.join(exports_dir, d))
    ]

    tokenizer = None
    label_encoder = None
    selected_model = None

    if not model_dirs:
        print("exports dizininde hiçbir model klasörü bulunamadı.")
        return None, None, None

    elif len(model_dirs) == 1:
        # Eğer yalnızca bir model klasörü varsa direkt onu seç
        selected_model = model_dirs[0]
        print(f"Sadece bir model bulundu: {selected_model}")

    else:
        # Birden fazla model klasörü varsa kullanıcıya sor
        print("Birden fazla model bulundu:")
        for idx, model_name in enumerate(model_dirs, 1):
            print(f"{idx}. {model_name}")
        try:
            choice = int(input("Bir model seçin (1, 2, ...): ")) - 1
            if 0 <= choice < len(model_dirs):
                selected_model = model_dirs[choice]
            else:
                print("Geçersiz seçim.")
                return None, None, None
        except ValueError:
            print("Geçersiz giriş. Lütfen sayı girin.")
            return None, None, None

    # Seçilen model dizininden tokenizer ve label encoder yolları
    selected_path = os.path.join(exports_dir, selected_model)
    tokenizer_path = os.path.join(selected_path, "tokenizer.pkl")
    label_encoder_path = os.path.join(selected_path, "label_encoder.pkl")

    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        print(f"Tokenizer yüklendi: {tokenizer_path}")
    else:
        print(f"Tokenizer bulunamadı: {tokenizer_path}")

    if os.path.exists(label_encoder_path):
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        print(f"Label encoder yüklendi: {label_encoder_path}")
    else:
        print(f"Label encoder bulunamadı: {label_encoder_path}")

    return tokenizer, label_encoder, selected_model

exports_dir = 'exports'
model_dir="lstm_model"
tokenizer, label_encoder,selected_model_dir = load_tokenizer_and_label_encoder(exports_dir)
    
   
model = load_model(os.path.join(model_dir,selected_model_dir))


user_input = input("Metni girin: ")

sequence = tokenizer.texts_to_sequences([user_input])  # Metni sayılara dönüştür
padded_sequence = pad_sequences(sequence, maxlen=maxlen)  # Padding işlemi

prediction = model.predict(padded_sequence)

predicted_class_index = np.argmax(prediction, axis=1)[0]
predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
category_name = newsgroups.target_names[predicted_label]




print(f"Tahmin edilen sınıf: {predicted_label}\n")
print(f"Tahmin edilen sınıf: {category_name}")