from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget,
    QAction, QFileDialog, QFrame, QSizePolicy, QStatusBar, QSpacerItem,
    QMenu, QInputDialog, QScrollArea, QApplication
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QCursor, QPalette, QColor
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from processing.filters import mean_filter, median_filter, edge_detection, sharpening_filter, smoothing_filter
from processing.histogram import show_histogram, histogram_equalization, contrast_stretching, contrast_spreading
from processing.geometry import rotate_image, flip_image
from processing.threshold import manual_threshold, otsu_threshold, kapur_threshold, local_threshold, adaptive_local_threshold
from processing.morphology import dilation, erosion
from processing.analysis import center_of_mass, mark_center_of_mass, zhang_suen_thinning

class MainWindow(QMainWindow):
    """
    Ana pencere sınıfı - Görüntü işleme uygulamasının ana arayüzünü oluşturur
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Görüntü İşleme Uygulaması")
        self.setWindowIcon(QIcon())  # İsterseniz buraya bir ikon dosyası ekleyebilirsiniz
        self.setMinimumSize(1200, 700)
        self.original_image = None
        self.processed_image = None
        
        # Tema modunu algıla ve uygula
        self.is_dark_mode = self.is_system_dark_mode()
        self.set_theme()

        # Menü
        open_action = QAction("Resim Aç", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Dosya")
        file_menu.addAction(open_action)

        # Durum çubuğu
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hoş geldiniz! Lütfen bir resim açın.", 5000)

        # Kenar çubuğu başlık
        sidebar_title = QLabel("İşlemler")
        sidebar_title.setFont(QFont("Arial", 16, QFont.Bold))
        sidebar_title.setAlignment(Qt.AlignCenter)
        sidebar_title.setStyleSheet(f"color: {self.get_title_color()}; margin-bottom: 15px;")

        # Butonlar (kenar çubuğu)
        self.create_buttons()

        # Kenar çubuğu düzeni ve Scroll Area
        scroll_widget = QWidget()
        self.button_layout = QVBoxLayout(scroll_widget)
        self.button_layout.setSpacing(12)  # Butonlar arası boşluğu artır
        self.button_layout.addWidget(sidebar_title)
        self.add_buttons_to_layout()
        self.button_layout.addStretch()
        
        # Scroll Area oluştur
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background: {self.get_sidebar_bg_color()};
                border-right: 2px solid {self.get_border_color()};
                border-radius: 0px;
            }}
            QScrollBar:vertical {{
                background: {self.get_scrollbar_bg_color()};
                width: 14px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {self.get_scrollbar_handle_color()};
                min-height: 20px;
                border-radius: 7px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

        # Görüntü alanları
        self.orig_label = QLabel()
        self.proc_label = QLabel()
        for label in [self.orig_label, self.proc_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setFrameShape(QFrame.Box)
            label.setStyleSheet(
                f"background: {self.get_image_bg_color()}; color: {self.get_text_color()}; "
                f"border: 2px solid {self.get_border_color()}; border-radius: 10px;"
            )
            label.setFont(QFont("Arial", 12, QFont.Bold))
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setMinimumSize(420, 420)
            label.setContextMenuPolicy(Qt.CustomContextMenu)
            label.customContextMenuRequested.connect(self.show_context_menu)

        # Alt açıklama metinleri
        orig_text = QLabel("Orijinal Görüntü")
        orig_text.setAlignment(Qt.AlignCenter)
        orig_text.setFont(QFont("Arial", 12, QFont.Bold))
        orig_text.setStyleSheet(f"color: {self.get_title_color()}; margin-top: 8px;")

        proc_text = QLabel("İşlenmiş Görüntü")
        proc_text.setAlignment(Qt.AlignCenter)
        proc_text.setFont(QFont("Arial", 12, QFont.Bold))
        proc_text.setStyleSheet(f"color: {self.get_title_color()}; margin-top: 8px;")

        # Görüntü ve açıklama düzeni
        orig_layout = QVBoxLayout()
        orig_layout.addWidget(self.orig_label)
        orig_layout.addWidget(orig_text)

        proc_layout = QVBoxLayout()
        proc_layout.addWidget(self.proc_label)
        proc_layout.addWidget(proc_text)

        image_layout = QHBoxLayout()
        image_layout.addLayout(orig_layout)
        image_layout.addSpacing(20)
        image_layout.addLayout(proc_layout)

        # Ana düzen: kenar çubuğu + görüntüler
        main_layout = QHBoxLayout()
        main_layout.addWidget(scroll_area, 1)
        main_layout.addSpacing(20)
        main_layout.addLayout(image_layout, 5)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
    
    def is_system_dark_mode(self):
        """Sistemin dark mode'da olup olmadığını kontrol et"""
        app = QApplication.instance()
        palette = app.palette()
        bg_color = palette.color(QPalette.Window)
        return bg_color.lightness() < 128  # Koyu arka planlar daha düşük lightness değerlerine sahiptir
    
    def set_theme(self):
        """Tema renklerini ayarla"""
        app = QApplication.instance()
        if self.is_dark_mode:
            app.setStyle("Fusion")
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(25, 25, 25))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, Qt.black)
            app.setPalette(palette)
    
    def get_title_color(self):
        """Başlık rengini tema moduna göre döndür"""
        return "#64b5f6" if self.is_dark_mode else "#1976d2"
    
    def get_border_color(self):
        """Kenarlık rengini tema moduna göre döndür"""
        return "#64b5f6" if self.is_dark_mode else "#1976d2"
    
    def get_sidebar_bg_color(self):
        """Kenar çubuğu arka plan rengini tema moduna göre döndür"""
        return "#2d2d30" if self.is_dark_mode else "#f5f5f5"
    
    def get_image_bg_color(self):
        """Görüntü arka planı rengini tema moduna göre döndür"""
        return "#1e1e1e" if self.is_dark_mode else "#222"
    
    def get_text_color(self):
        """Metin rengini tema moduna göre döndür"""
        return "#e0e0e0" if self.is_dark_mode else "#fff"
    
    def get_scrollbar_bg_color(self):
        """Kaydırma çubuğu arka plan rengini tema moduna göre döndür"""
        return "#2d2d30" if self.is_dark_mode else "#e0e0e0"
    
    def get_scrollbar_handle_color(self):
        """Kaydırma çubuğu tutamaç rengini tema moduna göre döndür"""
        return "#505050" if self.is_dark_mode else "#b0b0b0"
    
    def get_button_style(self):
        """Buton stilini tema moduna göre döndür"""
        if self.is_dark_mode:
            return """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #2c3e50, stop:1 #3498db);
                    color: white;
                    font-size: 16px;
                    padding: 14px;
                    border-radius: 10px;
                    margin-bottom: 3px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #34495e, stop:1 #2980b9);
                }
            """
        else:
            return """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #1976d2, stop:1 #64b5f6);
                    color: white;
                    font-size: 16px;
                    padding: 14px;
                    border-radius: 10px;
                    margin-bottom: 3px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #1565c0, stop:1 #42a5f5);
                }
            """
    
    def create_buttons(self):
        """Tüm butonları oluştur"""
        self.mean_btn = QPushButton("Ortalama (Mean) Filtresi")
        self.mean_btn.clicked.connect(self.apply_mean_filter)
        
        self.median_btn = QPushButton("Medyan (Median) Filtresi")
        self.median_btn.clicked.connect(self.apply_median_filter)
        
        self.edge_btn = QPushButton("Kenar Bulma Filtresi")
        self.edge_btn.clicked.connect(self.apply_edge_filter)
        
        self.sharp_btn = QPushButton("Keskinleştirme Filtresi")
        self.sharp_btn.clicked.connect(self.apply_sharpening_filter)
        
        self.smooth_btn = QPushButton("Yumuşatma Filtresi")
        self.smooth_btn.clicked.connect(self.apply_smoothing_filter)
        
        self.hist_btn = QPushButton("Histogram Göster")
        self.hist_btn.clicked.connect(self.show_histogram_window)
        
        self.histeq_btn = QPushButton("Histogram Eşitle")
        self.histeq_btn.clicked.connect(self.apply_histogram_equalization)
        
        self.contrast_stretch_btn = QPushButton("Kontrast Germe")
        self.contrast_stretch_btn.clicked.connect(self.apply_contrast_stretching)
        
        self.contrast_spread_btn = QPushButton("Kontrast Yayma")
        self.contrast_spread_btn.clicked.connect(self.apply_contrast_spreading)
        
        self.rotate90_btn = QPushButton("90° Döndür")
        self.rotate90_btn.clicked.connect(lambda: self.apply_rotate(90))
        
        self.rotate180_btn = QPushButton("180° Döndür")
        self.rotate180_btn.clicked.connect(lambda: self.apply_rotate(180))
        
        self.rotate270_btn = QPushButton("270° Döndür")
        self.rotate270_btn.clicked.connect(lambda: self.apply_rotate(270))
        
        self.flip_h_btn = QPushButton("Yatay Aynala")
        self.flip_h_btn.clicked.connect(lambda: self.apply_flip('horizontal'))
        
        self.flip_v_btn = QPushButton("Dikey Aynala")
        self.flip_v_btn.clicked.connect(lambda: self.apply_flip('vertical'))
        
        self.manual_thresh_btn = QPushButton("Manuel Eşikleme")
        self.manual_thresh_btn.clicked.connect(self.apply_manual_threshold)
        
        self.otsu_btn = QPushButton("OTSU Eşikleme")
        self.otsu_btn.clicked.connect(self.apply_otsu_threshold)
        
        self.kapur_btn = QPushButton("Kapur Eşikleme")
        self.kapur_btn.clicked.connect(self.apply_kapur_threshold)
        
        self.local_thresh_btn = QPushButton("Yerel Eşikleme (Blok)")
        self.local_thresh_btn.clicked.connect(self.apply_local_threshold)
        
        self.adaptive_thresh_btn = QPushButton("Adaptif Yerel Eşikleme")
        self.adaptive_thresh_btn.clicked.connect(self.apply_adaptive_local_threshold)
        
        self.dilate_btn = QPushButton("Dilation (Genişletme)")
        self.dilate_btn.clicked.connect(self.apply_dilation)
        
        self.erode_btn = QPushButton("Erosion (Aşındırma)")
        self.erode_btn.clicked.connect(self.apply_erosion)
        
        self.center_btn = QPushButton("Ağırlık Merkezi")
        self.center_btn.clicked.connect(self.apply_center_of_mass)
        
        self.skeleton_btn = QPushButton("İskelet Çıkar")
        self.skeleton_btn.clicked.connect(self.apply_skeleton)
        
        # Tüm butonlara stil uygula
        self.all_buttons = [
            self.mean_btn, self.median_btn, self.edge_btn, self.sharp_btn, self.smooth_btn,
            self.hist_btn, self.histeq_btn, self.contrast_stretch_btn, self.contrast_spread_btn,
            self.rotate90_btn, self.rotate180_btn, self.rotate270_btn,
            self.flip_h_btn, self.flip_v_btn, self.manual_thresh_btn, self.otsu_btn, self.kapur_btn,
            self.local_thresh_btn, self.adaptive_thresh_btn,
            self.dilate_btn, self.erode_btn, self.center_btn, self.skeleton_btn
        ]
        
        for btn in self.all_buttons:
            btn.setStyleSheet(self.get_button_style())
            btn.setMinimumHeight(50)  # Buton yüksekliğini artır
    
    def add_buttons_to_layout(self):
        """Butonları düzene ekle ve kategorilere ayır"""
        # Filtreler kategorisi
        self.add_category_title("Filtre İşlemleri")
        self.button_layout.addWidget(self.mean_btn)
        self.button_layout.addWidget(self.median_btn)
        self.button_layout.addWidget(self.edge_btn)
        self.button_layout.addWidget(self.sharp_btn)
        self.button_layout.addWidget(self.smooth_btn)
        
        # Histogram işlemleri
        self.add_category_title("Histogram İşlemleri")
        self.button_layout.addWidget(self.hist_btn)
        self.button_layout.addWidget(self.histeq_btn)
        self.button_layout.addWidget(self.contrast_stretch_btn)
        self.button_layout.addWidget(self.contrast_spread_btn)
        
        # Geometrik işlemler
        self.add_category_title("Geometrik İşlemler")
        self.button_layout.addWidget(self.rotate90_btn)
        self.button_layout.addWidget(self.rotate180_btn)
        self.button_layout.addWidget(self.rotate270_btn)
        self.button_layout.addWidget(self.flip_h_btn)
        self.button_layout.addWidget(self.flip_v_btn)
        
        # Eşikleme işlemleri
        self.add_category_title("Eşikleme İşlemleri")
        self.button_layout.addWidget(self.manual_thresh_btn)
        self.button_layout.addWidget(self.otsu_btn)
        self.button_layout.addWidget(self.kapur_btn)
        self.button_layout.addWidget(self.local_thresh_btn)
        self.button_layout.addWidget(self.adaptive_thresh_btn)
        
        # Morfolojik işlemler
        self.add_category_title("Morfolojik İşlemler")
        self.button_layout.addWidget(self.dilate_btn)
        self.button_layout.addWidget(self.erode_btn)
        self.button_layout.addWidget(self.center_btn)
        self.button_layout.addWidget(self.skeleton_btn)
    
    def add_category_title(self, title):
        """Kategori başlığı ekle"""
        label = QLabel(title)
        label.setFont(QFont("Arial", 14, QFont.Bold))
        label.setStyleSheet(f"color: {self.get_title_color()}; margin-top: 20px; margin-bottom: 10px;")
        self.button_layout.addWidget(label)

    def open_image(self):
        """
        Resim açma fonksiyonu - Kullanıcının bilgisayarından resim seçmesini sağlar
        """
        # Desteklenen formatları tanımla
        file_formats = (
            "Resim Dosyaları (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp);;"
            "PNG Dosyaları (*.png);;"
            "JPEG Dosyaları (*.jpg *.jpeg);;"
            "BMP Dosyaları (*.bmp);;"
            "TIFF Dosyaları (*.tif *.tiff);;"
            "GIF Dosyaları (*.gif);;"
            "WebP Dosyaları (*.webp);;"
            "Tüm Dosyalar (*.*)"
        )
        
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Resim Aç",
            "",
            file_formats
        )
        
        if fname:
            try:
                from PyQt5.QtGui import QImageReader
                reader = QImageReader(fname)
                qimg = reader.read()
                
                if not qimg.isNull():
                    qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
                    width = qimg.width()
                    height = qimg.height()
                    ptr = qimg.bits()
                    ptr.setsize(qimg.byteCount())
                    arr = np.array(ptr)
                    
                    # Kanal sayısını kontrol et
                    if arr.size == width * height * 3:
                        arr = arr.reshape(height, width, 3)
                        self.original_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    elif arr.size == width * height:
                        arr = arr.reshape(height, width)
                        self.original_image = arr
                    else:
                        self.status_bar.showMessage("Resim formatı desteklenmiyor!", 5000)
                        return
                        
                    self.processed_image = None
                    self.show_image(self.original_image, self.orig_label)
                    self.proc_label.clear()
                    
                    # Resim bilgilerini göster
                    image_info = f"Resim yüklendi. Boyut: {width}x{height} piksel"
                    self.status_bar.showMessage(image_info, 3000)
                    
                else:
                    self.status_bar.showMessage("Resim okunamadı! Lütfen geçerli bir resim dosyası seçin.", 5000)
                    
            except Exception as e:
                self.status_bar.showMessage(f"Hata oluştu: {str(e)}", 5000)

    def show_image(self, img, label):
        """
        Görüntüyü ekranda gösterme fonksiyonu
        """
        if img is not None:
            if len(img.shape) == 2:
                rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            label.clear()

    def apply_mean_filter(self):
        """
        Ortalama filtresi uygulama fonksiyonu - Gürültüyü azaltmak için kullanılır
        """
        if self.original_image is not None:
            result = mean_filter(self.original_image, kernel_size=3)
            self.processed_image = result
            self.show_image(result, self.proc_label)
            self.status_bar.showMessage("Ortalama filtresi uygulandı.", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)

    def apply_median_filter(self):
        """
        Medyan filtresi uygulama fonksiyonu - Tuz ve biber gürültüsünü gidermek için kullanılır
        """
        if self.original_image is not None:
            result = median_filter(self.original_image, kernel_size=3)
            self.processed_image = result
            self.show_image(result, self.proc_label)
            self.status_bar.showMessage("Medyan filtresi uygulandı.", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)

    def apply_edge_filter(self):
        """
        Kenar bulma filtresi uygulama fonksiyonu - Görüntüdeki kenarları tespit eder
        """
        if self.original_image is not None:
            result = edge_detection(self.original_image)
            self.processed_image = result
            self.show_image(result, self.proc_label)
            self.status_bar.showMessage("Kenar bulma filtresi uygulandı.", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)

    def apply_sharpening_filter(self):
        """
        Keskinleştirme filtresi uygulama fonksiyonu - Görüntüyü daha net hale getirir
        """
        if self.original_image is not None:
            result = sharpening_filter(self.original_image)
            self.processed_image = result
            self.show_image(result, self.proc_label)
            self.status_bar.showMessage("Keskinleştirme filtresi uygulandı.", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)

    def apply_smoothing_filter(self):
        """
        Yumuşatma filtresi uygulama fonksiyonu - Görüntüyü yumuşatır
        """
        if self.original_image is not None:
            result = smoothing_filter(self.original_image)
            self.processed_image = result
            self.show_image(result, self.proc_label)
            self.status_bar.showMessage("Yumuşatma filtresi uygulandı.", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)

    def show_histogram_window(self):
        """
        Histogram penceresini gösterme fonksiyonu - Görüntünün piksel dağılımını gösterir
        """
        if self.original_image is not None:
            show_histogram(self.original_image, "Orijinal Görüntü Histogramı")
            self.status_bar.showMessage("Histogram gösterildi.", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)

    def apply_histogram_equalization(self):
        """
        Histogram eşitleme fonksiyonu - Görüntünün kontrastını artırır
        """
        if self.original_image is not None:
            eq_img = histogram_equalization(self.original_image)
            self.processed_image = eq_img
            self.show_image(eq_img, self.proc_label)
            show_histogram(eq_img, "Eşitlenmiş Görüntü Histogramı")
            self.status_bar.showMessage("Histogram eşitleme uygulandı.", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)

    def apply_rotate(self, angle):
        """
        Görüntüyü döndürme fonksiyonu - Belirtilen açı kadar döndürür
        """
        if self.original_image is not None:
            rotated = rotate_image(self.original_image, angle)
            self.processed_image = rotated
            self.show_image(rotated, self.proc_label)
            self.status_bar.showMessage(f"{angle}° döndürme uygulandı.", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)

    def apply_flip(self, mode):
        """
        Görüntüyü aynalama fonksiyonu - Yatay veya dikey aynalama yapar
        """
        if self.original_image is not None:
            flipped = flip_image(self.original_image, mode)
            self.processed_image = flipped
            self.show_image(flipped, self.proc_label)
            self.status_bar.showMessage(f"{'Yatay' if mode=='horizontal' else 'Dikey'} aynalama uygulandı.", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)

    def apply_manual_threshold(self):
        """
        Manuel eşikleme fonksiyonu - Kullanıcının belirlediği eşik değerine göre ikili görüntü oluşturur
        """
        if self.original_image is not None:
            from PyQt5.QtWidgets import QInputDialog
            value, ok = QInputDialog.getInt(self, "Manuel Eşikleme", "Eşik değeri (0-255):", 128, 0, 255, 1)
            if ok:
                binary = manual_threshold(self.original_image, value)
                self.processed_image = binary
                self.show_image(binary, self.proc_label)
                self.status_bar.showMessage(f"Manuel eşikleme uygulandı. Eşik: {value}", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)

    def apply_otsu_threshold(self):
        """
        OTSU eşikleme fonksiyonu - Otomatik olarak en uygun eşik değerini belirler
        """
        if self.original_image is not None:
            binary = otsu_threshold(self.original_image)
            self.processed_image = binary
            self.show_image(binary, self.proc_label)
            self.status_bar.showMessage("OTSU eşikleme uygulandı.", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)

    def apply_kapur_threshold(self):
        """
        Kapur eşikleme fonksiyonu - Entropi tabanlı otomatik eşikleme yapar
        """
        if self.original_image is not None:
            binary = kapur_threshold(self.original_image)
            self.processed_image = binary
            self.show_image(binary, self.proc_label)
            self.status_bar.showMessage("Kapur eşikleme uygulandı.", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)

    def apply_local_threshold(self):
        """
        Yerel eşikleme fonksiyonu - Görüntüyü bloklara bölerek her blok için ayrı eşikleme yapar
        """
        if self.original_image is not None:
            # Kullanıcıdan parametreleri al
            block_size, ok1 = QInputDialog.getInt(
                self, "Blok Boyutu", "Blok boyutu (2-64):", 16, 2, 64, 2
            )
            if not ok1:
                return
                
            c_value, ok2 = QInputDialog.getInt(
                self, "C Değeri", "C değeri (0-20):", 5, 0, 20, 1
            )
            if not ok2:
                return
                
            # Yerel eşikleme uygula
            result = local_threshold(self.original_image, block_size, c_value)
            self.processed_image = result
            self.show_image(result, self.proc_label)
            self.status_bar.showMessage(f"Yerel eşikleme uygulandı. Blok: {block_size}x{block_size}, C: {c_value}", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)
            
    def apply_adaptive_local_threshold(self):
        """
        Adaptif yerel eşikleme fonksiyonu - Piksel bazlı adaptif eşikleme yapar
        """
        if self.original_image is not None:
            # Kullanıcıdan parametreleri al
            window_size, ok1 = QInputDialog.getInt(
                self, "Pencere Boyutu", "Pencere boyutu (tek sayı, 3-101):", 51, 3, 101, 2
            )
            # Tek sayı yap
            if window_size % 2 == 0:
                window_size += 1
                
            if not ok1:
                return
                
            c_value, ok2 = QInputDialog.getInt(
                self, "C Değeri", "C değeri (0-20):", 10, 0, 20, 1
            )
            if not ok2:
                return
                
            # Adaptif yerel eşikleme uygula
            result = adaptive_local_threshold(self.original_image, window_size, c_value)
            self.processed_image = result
            self.show_image(result, self.proc_label)
            self.status_bar.showMessage(f"Adaptif yerel eşikleme uygulandı. Pencere: {window_size}x{window_size}, C: {c_value}", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)

    def apply_dilation(self):
        """
        Dilation (genişletme) fonksiyonu - İkili görüntüdeki nesneleri genişletir
        """
        if self.processed_image is not None:
            dilated = dilation(self.processed_image, kernel_size=3)
            self.processed_image = dilated
            self.show_image(dilated, self.proc_label)
            self.status_bar.showMessage("Dilation uygulandı.", 3000)
        else:
            self.status_bar.showMessage("Önce bir ikili görüntü elde edin!", 3000)

    def apply_erosion(self):
        """
        Erosion (aşındırma) fonksiyonu - İkili görüntüdeki nesneleri küçültür
        """
        if self.processed_image is not None:
            eroded = erosion(self.processed_image, kernel_size=3)
            self.processed_image = eroded
            self.show_image(eroded, self.proc_label)
            self.status_bar.showMessage("Erosion uygulandı.", 3000)
        else:
            self.status_bar.showMessage("Önce bir ikili görüntü elde edin!", 3000)

    def apply_center_of_mass(self):
        """
        Ağırlık merkezi hesaplama fonksiyonu - İkili görüntüdeki nesnenin merkezini bulur
        """
        if self.processed_image is not None:
            center = center_of_mass(self.processed_image)
            marked = mark_center_of_mass(self.processed_image, center)
            self.show_image(marked, self.proc_label)
            self.status_bar.showMessage("Ağırlık merkezi işaretlendi.", 3000)
        else:
            self.status_bar.showMessage("Önce bir ikili görüntü elde edin!", 3000)

    def apply_skeleton(self):
        """
        İskelet çıkarma fonksiyonu - İkili görüntüdeki nesnenin iskeletini çıkarır
        """
        if self.processed_image is not None:
            # Önce resmi küçültün
            resized = cv2.resize(self.processed_image, (800, 600))
            # Sonra iskelet çıkarın
            skeleton = zhang_suen_thinning(resized)
            self.show_image(skeleton, self.proc_label)
            self.status_bar.showMessage("İskelet çıkarıldı.", 3000)
        else:
            self.status_bar.showMessage("Önce bir ikili görüntü elde edin!", 3000)

    def show_context_menu(self, position):
        # Hangi etiketin sağ tıklandığını belirle
        sender = self.sender()
        
        # Eğer görüntü yoksa menüyü gösterme
        if (sender == self.orig_label and self.original_image is None) or \
           (sender == self.proc_label and self.processed_image is None):
            return
            
        context_menu = QMenu()
        save_action = QAction("Görüntüyü Kaydet", self)
        context_menu.addAction(save_action)
        
        # Hangi görüntüyü kaydedeceğimizi belirle
        if sender == self.orig_label:
            save_action.triggered.connect(lambda: self.save_image(self.original_image))
        else:
            save_action.triggered.connect(lambda: self.save_image(self.processed_image))
            
        # Menüyü göster
        context_menu.exec_(QCursor.pos())

    def save_image(self, image):
        if image is None:
            self.status_bar.showMessage("Kaydedilecek görüntü bulunamadı!", 3000)
            return
            
        # Desteklenen formatları tanımla
        file_formats = (
            "PNG Dosyası (*.png);;"
            "JPEG Dosyası (*.jpg);;"
            "BMP Dosyası (*.bmp);;"
            "TIFF Dosyası (*.tif);;"
            "WebP Dosyası (*.webp)"
        )
        
        # Kaydetme yolunu al
        file_path, selected_format = QFileDialog.getSaveFileName(
            self,
            "Görüntüyü Kaydet",
            "",
            file_formats
        )
        
        if file_path:
            try:
                # Görüntüyü doğru formatta kaydet
                if len(image.shape) == 3:
                    # BGR'dan RGB'ye çevir
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    height, width, channel = rgb_image.shape
                    bytes_per_line = channel * width
                    qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                else:
                    # Gri tonlamalı görüntü
                    height, width = image.shape
                    qt_image = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
                
                # Görüntüyü kaydet
                if qt_image.save(file_path):
                    self.status_bar.showMessage(f"Görüntü başarıyla kaydedildi: {file_path}", 3000)
                else:
                    self.status_bar.showMessage("Görüntü kaydedilemedi!", 3000)
            except Exception as e:
                self.status_bar.showMessage(f"Kaydetme hatası: {str(e)}", 3000)

    def apply_contrast_stretching(self):
        if self.original_image is not None:
            # Kullanıcıdan parametre al
            min_out, ok1 = QInputDialog.getInt(
                self, "Min Değer", "Minimum çıkış değeri (0-255):", 0, 0, 255, 1
            )
            if not ok1:
                return
                
            max_out, ok2 = QInputDialog.getInt(
                self, "Max Değer", "Maksimum çıkış değeri (0-255):", 255, 0, 255, 1
            )
            if not ok2:
                return
                
            if min_out >= max_out:
                self.status_bar.showMessage("Hata: Min değer, Max değerden küçük olmalıdır!", 3000)
                return
                
            # Kontrast germe uygula
            result = contrast_stretching(self.original_image, min_out, max_out)
            self.processed_image = result
            self.show_image(result, self.proc_label)
            self.status_bar.showMessage(f"Kontrast germe uygulandı. Min: {min_out}, Max: {max_out}", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)
    
    def apply_contrast_spreading(self):
        if self.original_image is not None:
            # Kullanıcıdan yüzde değerini al
            percentage, ok = QInputDialog.getInt(
                self, "Yüzde Değeri", "Histogramdan kesilecek yüzde (1-20):", 5, 1, 20, 1
            )
            if not ok:
                return
                
            # Kontrast yayma uygula
            result = contrast_spreading(self.original_image, percentage)
            self.processed_image = result
            self.show_image(result, self.proc_label)
            self.status_bar.showMessage(f"Kontrast yayma uygulandı. Kırpma yüzdesi: %{percentage}", 3000)
        else:
            self.status_bar.showMessage("Önce bir resim yükleyin!", 3000)