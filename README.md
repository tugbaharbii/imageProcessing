# 🖼️ Görüntü İşleme Uygulaması

Bu proje, çeşitli görüntü işleme tekniklerini uygulayabilen kullanıcı dostu bir masaüstü uygulamasıdır. Görüntü işleme alanında yaygın olarak kullanılan birçok tekniği tek bir arayüzde sunmaktadır.

## ✨ Özellikler

### 🔍 Filtreler
- **Ortalama Filtresi**: Gürültü azaltma için kullanılan temel filtre
- **Medyan Filtresi**: Tuz ve biber gürültüsünü gidermek için etkili filtre
- **Kenar Bulma**: Görüntüdeki kenarları tespit etme
- **Keskinleştirme**: Görüntü detaylarını belirginleştirme
- **Yumuşatma**: Gürültüyü azaltma ve görüntüyü yumuşatma

### 📊 Histogram İşlemleri
- **Histogram Görüntüleme**: Piksel dağılımını analiz etme
- **Histogram Eşitleme**: Kontrastı iyileştirme
- **Kontrast Germe**: Görüntü kontrastını artırma
- **Kontrast Yayma**: Görüntü kontrastını dengeleme

### 📐 Geometrik İşlemler
- **Döndürme**: Görüntüyü istenilen açıda döndürme
- **Yatay/Dikey Aynalama**: Görüntüyü yatay veya dikey eksende aynalama

### ⚖️ Eşikleme İşlemleri
- **Manuel Eşikleme**: Kullanıcı tarafından belirlenen eşik değeri
- **OTSU Eşikleme**: Otomatik eşik değeri belirleme
- **Kapur Eşikleme**: Entropi tabanlı eşikleme
- **Yerel Eşikleme**: Bölgesel adaptif eşikleme
- **Adaptif Yerel Eşikleme**: Dinamik eşik değeri belirleme

### 🔬 Morfolojik İşlemler
- **Dilation (Genişletme)**: Nesneleri genişletme
- **Erosion (Aşındırma)**: Nesneleri küçültme

### 📈 Analiz İşlemleri
- **Ağırlık Merkezi Hesaplama**: Nesnelerin merkez noktasını bulma
- **İskelet Çıkarma**: Nesnelerin iskelet yapısını elde etme

## 🚀 Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamayı başlatın:
```bash
python main.py
```

## 💻 Kullanım

1. "Dosya Aç" butonu ile bir görüntü seçin
2. İstediğiniz işlemi menüden seçin
3. İşlenmiş görüntüyü kaydetmek için "Kaydet" butonunu kullanın

## 👥 Geliştiriciler

- **Kaan Şengün** - [GitHub Profili](https://github.com/Kaansengun)
- **Tuğba Harbi** - [GitHub Profili](https://github.com/tugbaharbii)


## 🤝 Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir özellik dalı oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Dalınıza push edin (`git push origin feature/amazing-feature`)
5. Bir Pull Request açın
