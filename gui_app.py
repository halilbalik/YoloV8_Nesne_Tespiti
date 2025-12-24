"""
BLG 407 - Makine Öğrenmesi
2. Proje Ödevi: YOLOv8 Nesne Tespiti - PyQt5 GUI Uygulaması

Öğrenci: Halil İbrahim Balık
Numara: 2212721046

"""
from ultralytics import YOLO
import sys
import os
import cv2
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QGroupBox,
    QStatusBar, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt


class YOLOv8DetectorApp(QMainWindow):
    """YOLOv8 Nesne Tespiti Uygulaması"""
    
    def __init__(self):
        super().__init__()
        
        # Model değişkenleri
        self.model = None
        self.model_path = "best.pt"
        self.current_image_path = None
        self.result_image = None
        self.detections = []
        
        # Sınıf isimleri
        self.class_names = ['cuzdan', 'saat']
        
        # Arayüzü başlat
        self.init_ui()
        
        # Modeli otomatik yükle (varsa)
        self.auto_load_model()
    
    def init_ui(self):
        """Kullanıcı arayüzünü oluştur"""

        # Pencere ayarları
        self.setWindowTitle("YOLOv8 Nesne Tespiti - Saat & Cüzdan")
        self.setGeometry(100, 100, 1200, 700)
        self.setMinimumSize(1000, 600)

        # Ana widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background-color: #f0f0f0;")

        # Ana layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Başlık
        title_label = QLabel("YOLOv8 Nesne Tespiti")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #000000; background-color: #ffffff; padding: 8px; border: 1px solid #c0c0c0;")
        main_layout.addWidget(title_label)

        # Alt başlık
        subtitle_label = QLabel("Saat ve Cüzdan Tespiti | BLG 407 - Halil İbrahim Balık")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setFont(QFont("Segoe UI", 9))
        subtitle_label.setStyleSheet("color: #606060; background-color: transparent; padding: 4px;")
        main_layout.addWidget(subtitle_label)
        
        # Görüntü panelleri container
        images_layout = QHBoxLayout()
        images_layout.setSpacing(10)

        # Sol panel - Orijinal Görüntü
        left_group = QGroupBox("Orijinal Görüntü")
        left_group.setFont(QFont("Segoe UI", 9))
        left_group.setStyleSheet("""
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 5px;
                background-color: #ffffff;
            }
        """)
        left_layout = QVBoxLayout(left_group)
        left_layout.setContentsMargins(8, 8, 8, 8)

        self.original_label = QLabel()
        self.original_label.setMinimumSize(450, 400)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("""
            QLabel {
                background-color: #fafafa;
                border: 1px solid #d0d0d0;
                color: #808080;
            }
        """)
        self.original_label.setText("Görüntü seçilmedi")
        left_layout.addWidget(self.original_label)

        images_layout.addWidget(left_group)

        # Sağ panel - Tespit Sonucu
        right_group = QGroupBox("Tespit Sonucu")
        right_group.setFont(QFont("Segoe UI", 9))
        right_group.setStyleSheet("""
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 5px;
                background-color: #ffffff;
            }
        """)
        right_layout = QVBoxLayout(right_group)
        right_layout.setContentsMargins(8, 8, 8, 8)

        self.result_label = QLabel()
        self.result_label.setMinimumSize(450, 400)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                background-color: #fafafa;
                border: 1px solid #d0d0d0;
                color: #808080;
            }
        """)
        self.result_label.setText("Tespit yapılmadı")
        right_layout.addWidget(self.result_label)

        images_layout.addWidget(right_group)

        main_layout.addLayout(images_layout)
        
        # Tespit sonuçları bilgi kutusu
        self.info_group = QGroupBox("Tespit Sonuçları")
        self.info_group.setFont(QFont("Segoe UI", 9))
        self.info_group.setStyleSheet("""
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 5px;
                background-color: #ffffff;
            }
        """)
        info_layout = QVBoxLayout(self.info_group)
        info_layout.setContentsMargins(8, 8, 8, 8)

        self.detection_info_label = QLabel("Henüz tespit yapılmadı.")
        self.detection_info_label.setAlignment(Qt.AlignCenter)
        self.detection_info_label.setFont(QFont("Segoe UI", 9))
        self.detection_info_label.setStyleSheet("padding: 8px; color: #404040;")
        info_layout.addWidget(self.detection_info_label)

        main_layout.addWidget(self.info_group)
        
        # Butonlar
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)

        # Model Seç butonu
        self.btn_load_model = QPushButton("Model Seç")
        self.btn_load_model.setMinimumHeight(32)
        self.btn_load_model.setFont(QFont("Segoe UI", 9))
        self.btn_load_model.setStyleSheet(self.get_button_style("standard"))
        self.btn_load_model.clicked.connect(self.load_model)
        buttons_layout.addWidget(self.btn_load_model)

        # Görüntü Seç butonu
        self.btn_select_image = QPushButton("Görüntü Seç")
        self.btn_select_image.setMinimumHeight(32)
        self.btn_select_image.setFont(QFont("Segoe UI", 9))
        self.btn_select_image.setStyleSheet(self.get_button_style("standard"))
        self.btn_select_image.clicked.connect(self.select_image)
        buttons_layout.addWidget(self.btn_select_image)

        # Tespit Et butonu
        self.btn_detect = QPushButton("Tespit Et")
        self.btn_detect.setMinimumHeight(32)
        self.btn_detect.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.btn_detect.setStyleSheet(self.get_button_style("primary"))
        self.btn_detect.clicked.connect(self.detect_objects)
        self.btn_detect.setEnabled(False)
        buttons_layout.addWidget(self.btn_detect)

        # Kaydet butonu
        self.btn_save = QPushButton("Kaydet")
        self.btn_save.setMinimumHeight(32)
        self.btn_save.setFont(QFont("Segoe UI", 9))
        self.btn_save.setStyleSheet(self.get_button_style("standard"))
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setEnabled(False)
        buttons_layout.addWidget(self.btn_save)

        # Temizle butonu
        self.btn_clear = QPushButton("Temizle")
        self.btn_clear.setMinimumHeight(32)
        self.btn_clear.setFont(QFont("Segoe UI", 9))
        self.btn_clear.setStyleSheet(self.get_button_style("standard"))
        self.btn_clear.clicked.connect(self.clear_all)
        buttons_layout.addWidget(self.btn_clear)

        main_layout.addLayout(buttons_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hazır. Lütfen bir model ve görüntü seçin.")
    
    def get_button_style(self, button_type):
        """Buton stili oluştur"""
        if button_type == "primary":
            # Tek vurgu rengi - Tespit Et butonu için
            return """
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    border: 1px solid #005a9e;
                    padding: 6px 16px;
                }
                QPushButton:hover {
                    background-color: #106ebe;
                }
                QPushButton:pressed {
                    background-color: #005a9e;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #888888;
                    border: 1px solid #b0b0b0;
                }
            """
        else:
            # Standart gri butonlar
            return """
                QPushButton {
                    background-color: #e1e1e1;
                    color: #000000;
                    border: 1px solid #adadad;
                    padding: 6px 16px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
                QPushButton:pressed {
                    background-color: #c0c0c0;
                }
                QPushButton:disabled {
                    background-color: #f0f0f0;
                    color: #a0a0a0;
                    border: 1px solid #d0d0d0;
                }
            """
    
    def auto_load_model(self):
        """Otomatik model yükleme"""
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                self.status_bar.showMessage(f"Model otomatik yüklendi: {self.model_path}")
            except Exception as e:
                self.status_bar.showMessage(f"Uyarı: Model yüklenemedi - {str(e)}")
        else:
            self.status_bar.showMessage("best.pt bulunamadı. Lütfen model seçin.")
    
    def load_model(self):
        """Model dosyası seç ve yükle"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "YOLOv8 Model Seç",
            "",
            "PyTorch Model (*.pt);;Tüm Dosyalar (*.*)"
        )

        if file_path:
            try:
                self.model = YOLO(file_path)
                self.model_path = file_path
                self.status_bar.showMessage(f"Model yüklendi: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Başarılı", f"Model yüklendi:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Model yüklenemedi:\n{str(e)}")
    
    def select_image(self):
        """Görüntü seç"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Görüntü Seç",
            "",
            "Görüntü Dosyaları (*.jpg *.jpeg *.png *.bmp);;Tüm Dosyalar (*.*)"
        )

        if file_path:
            self.current_image_path = file_path

            # Orijinal görüntüyü göster
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                self.original_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.original_label.setPixmap(scaled_pixmap)

            # Sonuç panelini temizle
            self.result_label.clear()
            self.result_label.setText("Tespit için 'Tespit Et' butonuna tıklayın")
            self.detection_info_label.setText("Tespit bekleniyor...")

            # Butonları aktif et
            self.btn_detect.setEnabled(True)
            self.btn_save.setEnabled(False)

            self.status_bar.showMessage(f"Görüntü yüklendi: {os.path.basename(file_path)}")
    
    def detect_objects(self):
        """Nesne tespiti yap"""
        if self.model is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir model yükleyin!")
            return

        if self.current_image_path is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir görüntü seçin!")
            return

        try:
            self.status_bar.showMessage("Tespit yapılıyor...")
            QApplication.processEvents()

            # Tahmin yap
            results = self.model.predict(
                self.current_image_path,
                conf=0.25,  # Confidence threshold
                verbose=False
            )

            # Sonuç görüntüsünü al
            result = results[0]
            self.result_image = result.plot()  # BGR formatında

            # Tespitleri kaydet
            self.detections = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                self.detections.append({
                    'class': cls_name,
                    'confidence': conf
                })

            # BGR -> RGB dönüşümü
            result_rgb = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)

            # QImage'e dönüştür
            h, w, ch = result_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(result_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Göster
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.result_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.result_label.setPixmap(scaled_pixmap)

            # Tespit bilgilerini güncelle
            self.update_detection_info()

            # Kaydet butonunu aktif et
            self.btn_save.setEnabled(True)

            self.status_bar.showMessage(f"Tespit tamamlandı - {len(self.detections)} nesne bulundu")

        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Tespit sırasında hata oluştu:\n{str(e)}")
            self.status_bar.showMessage(f"Hata: {str(e)}")
    
    def update_detection_info(self):
        """Tespit bilgilerini güncelle"""
        if not self.detections:
            self.detection_info_label.setText("Hiçbir nesne tespit edilemedi.")
            return

        # Sınıf sayılarını hesapla
        class_counts = {}
        for det in self.detections:
            cls = det['class']
            if cls not in class_counts:
                class_counts[cls] = {'count': 0, 'confidences': []}
            class_counts[cls]['count'] += 1
            class_counts[cls]['confidences'].append(det['confidence'])

        # Bilgi metnini oluştur
        info_text = f"<b>Toplam Tespit:</b> {len(self.detections)} nesne<br><br>"

        for cls_name, data in class_counts.items():
            avg_conf = sum(data['confidences']) / len(data['confidences'])
            info_text += f"<b>{cls_name.upper()}</b>: {data['count']} adet (Ortalama güven: %{avg_conf*100:.1f})<br>"

        self.detection_info_label.setText(info_text)
    
    def save_result(self):
        """Sonuç görüntüsünü kaydet"""
        if self.result_image is None:
            QMessageBox.warning(self, "Uyarı", "Kaydedilecek sonuç yok!")
            return

        # Varsayılan dosya adı
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"detection_result_{timestamp}.jpg"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Sonucu Kaydet",
            default_name,
            "JPEG (*.jpg);;PNG (*.png);;Tüm Dosyalar (*.*)"
        )

        if file_path:
            try:
                cv2.imwrite(file_path, self.result_image)
                self.status_bar.showMessage(f"Kaydedildi: {file_path}")
                QMessageBox.information(self, "Başarılı", f"Görüntü kaydedildi:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Kaydetme hatası:\n{str(e)}")
    
    def clear_all(self):
        """Tümünü temizle"""
        self.current_image_path = None
        self.result_image = None
        self.detections = []

        self.original_label.clear()
        self.original_label.setText("Görüntü seçilmedi")

        self.result_label.clear()
        self.result_label.setText("Tespit yapılmadı")

        self.detection_info_label.setText("Henüz tespit yapılmadı.")

        self.btn_detect.setEnabled(False)
        self.btn_save.setEnabled(False)

        self.status_bar.showMessage("Temizlendi. Yeni bir görüntü seçebilirsiniz.")


def main():
    """Ana fonksiyon"""
    app = QApplication(sys.argv)
    
    # Uygulama stili
    app.setStyle('Fusion')
    
    # Ana pencere
    window = YOLOv8DetectorApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
