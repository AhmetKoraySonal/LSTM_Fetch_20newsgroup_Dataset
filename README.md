LSTM ile Metin Sınıflandırma Projesi
📌 Proje Açıklaması
Bu proje, 20 Newsgroups veri seti üzerinde LSTM (Long Short-Term Memory) ağları kullanarak metin sınıflandırma yapan bir derin öğrenme modeli içermektedir. Model, farklı haber gruplarına ait metinleri otomatik olarak sınıflandırabilir.

🛠 Teknik Detaylar

⚙️ Hiperparametreler
python
--epochs 200          # Eğitim iterasyon sayısı
--patience 5          # Erken durdurma için beklenen epoch
--batch_size 32       # Batch boyutu
--num_words 10000     # Tokenizer'ın kullanacağı max kelime sayısı
--maxlen 100          # Padding için maksimum metin uzunluğu
--output_dim 32       # Embedding boyutu
--lstm_units 32       # LSTM hücre sayısı


📂 Dosya Yapısı
.
├── config.py         # Hiperparametre yapılandırması
├── train.py          # Model eğitim scripti
├── test.py           # Model test scripti
├── exports/          # Eğitilmiş tokenizer ve label encoder
│   └── model_XX/     # Model versiyonuna göre klasör
│       ├── tokenizer.pkl
│       └── label_encoder.pkl
├── lstm_model/       # Eğitilmiş modeller
│   └── model_XX.h5   # Model dosyaları
└── Plots/            # Eğitim grafikleri


🚀 Kullanım


🔧 Eğitim
bash
python train.py [--epochs 200] [--batch_size 32] [--maxlen 100] ...


🔍 Test
bash
python test.py
Not: Test öncesi exports ve lstm_model klasörlerinde ilgili model dosyalarının bulunduğundan emin olun.

📊 Performans Metrikleri
Accuracy: Sınıflandırma doğruluğu

F1 Score: Precision ve recall dengesi

Loss: Eğitim ve validasyon kaybı

📈 Görselleştirme
Eğitim sonunda otomatik olarak üretilen grafikler:

Eğitim ve validasyon kaybı

Eğitim ve validasyon doğruluğu

💾 Kayıt Sistemi
Modeller versiyon kontrolü ile kaydedilir (model_01.h5, model_02.h5, ...)

Her model için log bilgileri log.txt'de tutulur

Tokenizer ve label encoder ilgili model klasörüne kaydedilir

🧪 Test Senaryosu
python
# Örnek çalıştırma:
Metni girin: "The new graphics card from NVIDIA shows amazing performance in latest games"
Tahmin edilen sınıf: 7 (comp.graphics)
