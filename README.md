# 🧠 TTG5 Hackathon 2025 - Medikal Göz Pedi Kalite Kontrol Sistemi

Bu proje, **Turgutlu Teknoloji Günleri 2025** kapsamında geliştirilen bir görüntü işleme ve makine öğrenmesi tabanlı sistemdir. Amaç, üretim hattında ilerleyen medikal göz pedlerinin **leke, yırtık, renk farkı** gibi kusurlarını tespit edip sınıflandırarak hatalı ürünleri sağlamlardan ayırmaktır.

## 📁 Proje Yapısı

```
gozPediFastApi/         → REST API servis dosyaları
runs_v2/                → YOLOv8 tahmin çıktı klasörü
Unity/                  → Unity simülasyon projesi
iOS/                    → iOS mobil uygulama dosyaları
hackathon.ipynb         → Eğitim ve test Jupyter defteri
yolov8_model.pt         → YOLOv8 segmentasyon modeli
eye_pad_model.keras     → Keras sınıflandırma modeli
```

---

## ⚙️ Kurulum ve Çalıştırma

### 1. Sanal Ortam Kurulumu

```bash
python -m venv venv
source venv/bin/activate      # Windows için: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API'yi Başlat

```bash
cd gozPediFastApi
uvicorn main:app --reload
```

> API ile modele görseller gönderip JSON olarak sonuç alabilirsiniz.

### 3. Eğitim/Deneme (Opsiyonel)

```bash
jupyter notebook
```

`hackathon.ipynb` dosyasını çalıştırarak modellerin eğitim/test sürecini gözlemleyebilirsiniz.

---

## 📱 iOS Mobil Uygulama

* `iOS/` klasörü, Swift diliyle geliştirilmiş olan iOS uygulama dosyalarını içerir.
* Uygulama, cihaz kamerası ile fotoğraf çeker ve bu görseli FastAPI sunucusuna gönderir.
* API'den gelen sınıflandırma sonucunu kullanıcıya anlık olarak sunar.
* Kullanıcılar hata oranı eşik değerini mobil arayüzden belirleyebilir.

> Mobil uygulama, aynı ağda çalışan API servisine HTTP üzerinden bağlanarak gerçek zamanlı analiz yapılmasını sağlar.

---

## 🎮 Simülasyon Ortamı (Unity)

1. Unity Hub üzerinden `Unity/` klasörünü açın.
2. Bant üzerinde ürün akışını ve hatalı/sorunsuz ayrımını simüle eden sahneyi çalıştırın.
3. Her ürünün rengine veya segmentasyon maskesine göre sınıflandırıldığını görebilirsiniz.

---

## 🧠 Kullanılan Modeller ve Karmaşıklık Analizi

### 1. YOLOv8 Segmentasyon Modeli

* **Amaç:** Göz pedlerini tespit etme ve maskeyle segmentasyon.
* **Zaman Karmaşıklığı:** O(n)
* **Mekan Karmaşıklığı:** O(n)

### 2. Keras CNN Modeli

* **Amaç:** Segmentasyondan sonra her pedin yapısal hatasını sınıflandırmak.
* **Zaman Karmaşıklığı:** O(n \* k^2)
* **Mekan Karmaşıklığı:** O(n)

---

## ⏱ FPS Testi (Gerçek Zamanlılık)

Mobil kamera ile yapılan testlerde sistem **1 dakikada ortalama 43** görseli işleyebilmektedir.

---

## 📌 Demo

Projeye ait demo videosu [demo.mp4](demo_link_here) olarak eklenmelidir.

---

## 💎 GitHub Etiketi

```
#ttg5hackathon2025
```

---

## 📊 Requirements

Proje bağımlılıkları için `requirements.txt` dosyasına aşağıdakiler eklenmelidir:

```txt
tensorflow>=2.9.0
keras>=2.9.0
ultralytics>=8.0.0
opencv-python
matplotlib
fastapi
uvicorn
numpy
pillow
```
## 👥 Ekip Üyeleri

- Bilal Dalgıç – Görüntü İşleme ve Backend Geliştirici
- Sefacan Demir – Mobil Uygulama Geliştirici (iOS)
- Zeynep Gülten – Simülasyon ve Unity Tasarımı
- Mehmet Altay Erdem – Veri Ön İşleme ve Model Eğitimi

