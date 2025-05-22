# 🧠 TTG5 Hackathon 2025 - Medikal Göz Pedi Kalite Kontrol Sistemi #ttg5hackathon2025

Bu proje, **Turgutlu Teknoloji Günleri 2025** kapsamında geliştirilen bir görüntü işleme ve makine öğrenmesi tabanlı sistemdir. Amaç, üretim hattında ilerleyen medikal göz pedlerinin **leke, yırtık, renk farkı** gibi kusurlarını tespit edip sınıflandırarak hatalı ürünleri sağlamlardan ayırmaktır.  

## 📁 Proje Yapısı

```
gozPediFastApi/         → REST API servis dosyaları
runs_v2/                → YOLOv8 tahmin çıktı klasörü
Unity/                  → Unity simülasyon projesi
EyePadRecognizer2 11/   → iOS mobil uygulama dosyaları
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

* `EyePadRecognizer2 11/` klasörü, Swift diliyle geliştirilmiş olan iOS uygulama dosyalarını içerir.
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

## 🧮 Algoritma Karmaşıklığı – Big-O Notasyonu

Aşağıda projede kullanılan tüm modüllerin zaman ve mekân (bellek) karmaşıklıkları Big-O notasyonu ile verilmiştir:

| Bileşen / Algoritma                     | Zaman Karmaşıklığı     | Mekân Karmaşıklığı     | Açıklama |
|----------------------------------------|------------------------|------------------------|----------|
| YOLOv8 Segmentasyon                    | O(n)                  | O(m)                  | n: piksel sayısı (640x640), m: model ağırlıkları + geçici bellek |
| `cv2.fitEllipse()`                     | O(k)                  | O(1)                  | k: kontur noktası sayısı |
| `get_ellipse_point()` (uç nokta)       | O(1)                  | O(1)                  | Sabit trigonometrik işlemler |
| Noktalar arası uzaklık (`np.linalg.norm`) | O(1)               | O(1)                  | İki nokta arası mesafe |
| JSON raporlama                         | O(n)                  | O(n)                  | n: analiz edilen yön sayısı |
| 📦 **Keras Leke Sınıflandırma**        |                        |                        |          |
| `ImageDataGenerator()`                | O(1)                  | O(1)                  | Parametre tanımı |
| `flow_from_directory()`               | O(n)                  | O(n)                  | n: klasördeki görsel sayısı |
| `MobileNetV2` (özellik çıkarımı)      | O(f)                  | O(f)                  | f: sabit FLOP (~3.5M parametre) |
| `GlobalAveragePooling2D()`            | O(h×w×c)              | O(c)                  | Sabit çıktı boyutu |
| `Dense(2, softmax)` katmanı           | O(c)                  | O(c)                  | c: giriş boyutu |
| `Dropout(0.5)` (eğitimde)             | O(c)                  | O(c)                  | Eğitim sırasında maskeleme |
| `model.fit()` (15 epoch)              | O(e·n·f)              | O(b·f)                | e: epoch, n: örnek sayısı, b: batch size |
| `model.predict(img)`                  | O(f)                  | O(f)                  | Tek görsel tahmini |

---

### 🧾 Terimler:

- `n`: Görsel sayısı (veri kümesi boyutu)
- `k`: Kontur noktası sayısı
- `m`: YOLO model parametreleri
- `f`: MobileNetV2 hesaplama büyüklüğü (sabit ≈ 3.5M parametre)
- `e`: Epoch sayısı (örnek: 15)
- `b`: Batch size (örnek: 16)
- `h×w×c`: CNN'den çıkan son katmanın boyutu (sabit)

---

## 📌 Demo

Projeye ait demo videosu: [video.mp4](./video.mp4)

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

