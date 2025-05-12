LSTM ile Metin Sınıflandırma Projesi
📌 Proje Açıklaması
Bu proje, 20 Newsgroups veri seti üzerinde LSTM (Long Short-Term Memory) ağları kullanarak metin sınıflandırma yapan bir derin öğrenme modeli içermektedir. Model, farklı haber gruplarına ait metinleri otomatik olarak sınıflandırabilir.


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
