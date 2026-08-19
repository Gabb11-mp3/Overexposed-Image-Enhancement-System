import math
import sys
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    return cv2.erode(dc, kernel)


def AtmLight(im, dark):
    h, w = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)
    indices = darkvec.argsort()[imsz - numpx :]
    atmsum = np.zeros((1, 3))

    for ind in range(numpx):
        atmsum += imvec[indices[ind]]

    atmospheric_light = atmsum / numpx
    return atmospheric_light / max(float(np.max(atmospheric_light)), 1e-8)


def TransmissionEstimate(im, atmospheric_light, sz):
    omega = 0.95
    normalized = np.empty(im.shape, im.dtype)
    for ind in range(3):
        normalized[:, :, ind] = im[:, :, ind] / max(
            float(atmospheric_light[0, ind]), 1e-8
        )
    return 1 - omega * DarkChannel(normalized, sz)


def Guidedfilter(im, p, r, eps):
    mean_i = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    covariance = mean_ip - mean_i * mean_p
    mean_ii = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    variance = mean_ii - mean_i * mean_i
    a = covariance / (variance + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    return mean_a * im + mean_b


def TransmissionRefine(im, estimated_transmission):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
    return Guidedfilter(gray, estimated_transmission, 60, 0.0001)


def Recover(im, transmission, atmospheric_light, tx=0.1):
    result = np.empty(im.shape, im.dtype)
    transmission = cv2.max(transmission, tx)
    for ind in range(3):
        result[:, :, ind] = (
            (im[:, :, ind] - atmospheric_light[0, ind]) / transmission
            + atmospheric_light[0, ind]
        )
    return np.clip(result, 0, 1)


def enhance_image(file_path, target_size=(500, 500)):
    source = cv2.imread(file_path)
    if source is None:
        raise FileNotFoundError(
            f"Image file '{file_path}' was not found or could not be loaded."
        )

    height, width = source.shape[:2]
    scale_factor = min(target_size[1] / height, target_size[0] / width)
    new_size = (int(width * scale_factor), int(height * scale_factor))
    resized_source = cv2.resize(source, new_size, interpolation=cv2.INTER_AREA)
    normalized_source = resized_source.astype(np.float64) / 255

    dark = DarkChannel(normalized_source, 15)
    atmospheric_light = AtmLight(normalized_source, dark)
    estimated_transmission = TransmissionEstimate(
        normalized_source, atmospheric_light, 15
    )
    transmission = TransmissionRefine(resized_source, estimated_transmission)
    recovered = Recover(
        normalized_source, transmission, atmospheric_light, 0.1
    )
    recovered = (recovered * 255).astype(np.uint8)

    filtered = cv2.bilateralFilter(
        recovered, d=2, sigmaColor=80, sigmaSpace=80
    )
    denoised = cv2.fastNlMeansDenoisingColored(
        filtered, None, 3, 3, 7, 15
    )
    blended = cv2.addWeighted(filtered, 0.9, denoised, 0.1, 0)
    detail = cv2.subtract(
        filtered, cv2.GaussianBlur(filtered, (5, 5), 2.0)
    )
    detail = cv2.addWeighted(detail, 1, blended, 1, 0)
    gaussian_blurred = cv2.GaussianBlur(detail, (7, 7), 1.5)
    enhanced = cv2.addWeighted(detail, 1.5, gaussian_blurred, -0.5, 0)

    return resized_source, enhanced


def calculate_entropy(image):
    histogram, _ = np.histogram(
        image.flatten(), bins=256, range=(0, 256), density=True
    )
    histogram = histogram[histogram > 0]
    return float(-np.sum(histogram * np.log2(histogram)))


def calculate_colorfulness(image):
    blue, green, red = cv2.split(image.astype(float))
    red_green = red - green
    yellow_blue = 0.5 * (red + green) - blue
    return float(
        np.hypot(np.std(red_green), np.std(yellow_blue))
        + 0.3 * np.hypot(np.mean(red_green), np.mean(yellow_blue))
    )


def calculate_metrics(original, enhanced):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    return (
        calculate_entropy(original_gray),
        calculate_entropy(enhanced_gray),
        calculate_colorfulness(original),
        calculate_colorfulness(enhanced),
    )


class ImageEnhancerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.metrics_data = []
        self.enhanced_image = None
        self.current_file_path = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Sistem Pemulihan Imej")
        self.resize(1280, 780)
        self.setMinimumSize(920, 640)

        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(32, 26, 32, 24)
        main_layout.setSpacing(22)

        # Header and primary actions
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(3)
        eyebrow_label = QLabel("PEMULIHAN IMEJ DIGITAL")
        eyebrow_label.setObjectName("eyebrowLabel")
        title_label = QLabel("Sistem Pemulihan Imej")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel(
            "Pulihkan butiran dan tingkatkan kualiti imej berkeamatan tinggi."
        )
        subtitle_label.setObjectName("subtitleLabel")
        title_layout.addWidget(eyebrow_label)
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)

        self.load_button = QPushButton("Buka Imej Input")
        self.load_button.setObjectName("secondaryButton")
        self.load_button.setCursor(Qt.PointingHandCursor)
        self.load_button.clicked.connect(self.load_image)

        self.save_button = QPushButton("Simpan Imej Output")
        self.save_button.setObjectName("primaryButton")
        self.save_button.setCursor(Qt.PointingHandCursor)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_image)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        header_layout.addWidget(self.load_button)
        header_layout.addWidget(self.save_button)
        main_layout.addLayout(header_layout)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.HLine)
        main_layout.addWidget(divider)

        # Reusable image card keeps the input and output visually consistent.
        def create_image_card(step, title, description, accent):
            card = QFrame()
            card.setObjectName("imageCard")
            card.setProperty("accent", accent)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(20, 18, 20, 20)
            card_layout.setSpacing(14)

            card_header = QHBoxLayout()
            card_header.setSpacing(12)

            card_title_layout = QVBoxLayout()
            card_title_layout.setSpacing(2)
            step_label = QLabel(step)
            step_label.setObjectName("stepLabel")
            card_title = QLabel(title)
            card_title.setObjectName("cardTitle")
            card_description = QLabel(description)
            card_description.setObjectName("cardDescription")
            card_title_layout.addWidget(step_label)
            card_title_layout.addWidget(card_title)
            card_title_layout.addWidget(card_description)

            state_badge = QLabel("MENUNGGU")
            state_badge.setObjectName("stateBadge")
            state_badge.setAlignment(Qt.AlignCenter)

            card_header.addLayout(card_title_layout)
            card_header.addStretch()
            card_header.addWidget(state_badge, 0, Qt.AlignTop)

            image_label = QLabel()
            image_label.setObjectName("imagePreview")
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumSize(360, 360)
            image_label.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding
            )

            card_layout.addLayout(card_header)
            card_layout.addWidget(image_label, 1)
            return card, image_label, state_badge

        image_layout = QHBoxLayout()
        image_layout.setSpacing(18)

        input_card, self.original_label, self.input_badge = create_image_card(
            "LANGKAH 01",
            "Imej Input",
            "Imej asal yang dipilih",
            "input",
        )
        self.original_label.setText(
            "Tiada imej dipilih\n\nKlik ‘Buka Imej Input’ untuk bermula"
        )

        output_card, self.enhanced_label, self.output_badge = create_image_card(
            "LANGKAH 02",
            "Imej Output",
            "Hasil selepas proses pemulihan",
            "output",
        )
        self.enhanced_label.setText(
            "Hasil pemulihan akan dipaparkan di sini"
        )

        image_layout.addWidget(input_card, 1)
        image_layout.addWidget(output_card, 1)
        main_layout.addLayout(image_layout, 1)

        # Persistent status area gives clear feedback without interrupting work.
        status_frame = QFrame()
        status_frame.setObjectName("statusFrame")
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(16, 11, 16, 11)
        status_layout.setSpacing(10)
        status_dot = QLabel("●")
        status_dot.setObjectName("statusDot")
        self.status_label = QLabel("Muat naik imej untuk dipulihkan")
        self.status_label.setObjectName("statusLabel")
        status_tip = QLabel("PNG · JPG · JPEG · BMP")
        status_tip.setObjectName("statusTip")
        status_layout.addWidget(status_dot)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(status_tip)
        main_layout.addWidget(status_frame)

        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)
        file_menu = menu_bar.addMenu("Fail")

        load_action = QAction("Muat naik Imej Input", self)
        load_action.triggered.connect(self.load_image)
        self.save_action = QAction("Simpan Imej Output", self)
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self.save_image)

        file_menu.addAction(load_action)
        file_menu.addAction(self.save_action)

        self.setStyleSheet(
            """
            * {
                font-family: "Segoe UI";
                color: #E8ECF7;
            }
            QMainWindow, QWidget#centralWidget {
                background-color: #0B1020;
            }
            QMenuBar {
                background-color: #0B1020;
                color: #AAB3CC;
                padding: 5px 24px;
                border-bottom: 1px solid #202A43;
            }
            QMenuBar::item {
                padding: 6px 10px;
                border-radius: 5px;
            }
            QMenuBar::item:selected, QMenu {
                background-color: #182138;
            }
            QMenu {
                border: 1px solid #2B3858;
                padding: 5px;
            }
            QLabel#eyebrowLabel {
                color: #7D8EFF;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#titleLabel {
                color: #FFFFFF;
                font-size: 27px;
                font-weight: 700;
            }
            QLabel#subtitleLabel {
                color: #8792AE;
                font-size: 12px;
            }
            QFrame#divider {
                color: #202A43;
                background-color: #202A43;
                max-height: 1px;
                border: none;
            }
            QPushButton {
                min-height: 42px;
                padding: 0 20px;
                border-radius: 9px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton#primaryButton {
                background-color: #6878F7;
                color: white;
                border: 1px solid #7C8AFF;
            }
            QPushButton#primaryButton:hover {
                background-color: #7B89FF;
            }
            QPushButton#secondaryButton {
                background-color: #151D31;
                color: #DCE2F2;
                border: 1px solid #2B3858;
            }
            QPushButton#secondaryButton:hover {
                background-color: #1D2943;
                border-color: #46577D;
            }
            QPushButton:disabled {
                background-color: #20283A;
                color: #626D87;
                border-color: #29334A;
            }
            QFrame#imageCard {
                background-color: #12192A;
                border: 1px solid #26324C;
                border-radius: 14px;
            }
            QFrame#imageCard[accent="output"] {
                border-color: #2D564E;
            }
            QLabel#stepLabel {
                color: #73809F;
                font-size: 9px;
                font-weight: 700;
            }
            QLabel#cardTitle {
                color: #F7F9FF;
                font-size: 17px;
                font-weight: 650;
            }
            QLabel#cardDescription {
                color: #7F8AA5;
                font-size: 11px;
            }
            QLabel#stateBadge {
                background-color: #202A41;
                color: #93A0BE;
                border: 1px solid #303C5A;
                border-radius: 8px;
                padding: 5px 9px;
                font-size: 9px;
                font-weight: 700;
            }
            QLabel#imagePreview {
                background-color: #090E1A;
                color: #68748F;
                border: 1px dashed #35415F;
                border-radius: 10px;
                font-size: 12px;
                padding: 12px;
            }
            QFrame#statusFrame {
                background-color: #111829;
                border: 1px solid #26314A;
                border-radius: 10px;
            }
            QLabel#statusDot {
                color: #4DD7A4;
                font-size: 11px;
            }
            QLabel#statusLabel {
                color: #AAB4CE;
                font-size: 11px;
            }
            QLabel#statusTip {
                color: #66728E;
                font-size: 10px;
                font-weight: 600;
            }
            QToolTip {
                background-color: #182138;
                color: white;
                border: 1px solid #34415F;
                padding: 6px;
            }
            """
        )

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Buka Imej Input",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if file_path:
            self.display_images(file_path)

    def display_images(self, file_path):
        self.load_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.save_action.setEnabled(False)
        self.input_badge.setText("MEMUAT")
        self.output_badge.setText("MEMPROSES")
        self.enhanced_label.clear()
        self.enhanced_label.setText("Imej sedang dipulihkan…")
        self.status_label.setText("Sedang memproses imej, sila tunggu…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            original_image, enhanced_image = enhance_image(file_path)
            self.current_file_path = file_path
            self.enhanced_image = enhanced_image

            self.original_label.setPixmap(
                self.convert_cv_to_pixmap(original_image)
            )
            self.original_label.setToolTip(f"Imej Input: {file_path}")
            self.enhanced_label.setPixmap(
                self.convert_cv_to_pixmap(enhanced_image)
            )
            self.enhanced_label.setToolTip(f"Imej Output: {file_path}")
            self.input_badge.setText("DIMUAT")
            self.output_badge.setText("SIAP")
            self.save_button.setEnabled(True)
            self.save_action.setEnabled(True)

            metrics = calculate_metrics(original_image, enhanced_image)
            self.status_label.setText(
                f"Entropi Input: {metrics[0]:.4f}, "
                f"Entropi Output: {metrics[1]:.4f}   "
                f"CI Input: {metrics[2]:.4f}, "
                f"CI Output: {metrics[3]:.4f}"
            )
            self.save_metrics_to_excel(file_path, metrics)
        except Exception as error:
            self.enhanced_image = None
            self.input_badge.setText("RALAT")
            self.output_badge.setText("GAGAL")
            self.enhanced_label.clear()
            self.enhanced_label.setText("Pemulihan imej tidak berjaya")
            self.status_label.setText(f"Ralat: {error}")
        finally:
            QApplication.restoreOverrideCursor()
            self.load_button.setEnabled(True)

    def save_image(self):
        if self.enhanced_image is None:
            self.status_label.setText("Tiada imej output untuk disimpan.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Simpan Imej Output",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not save_path:
            return

        if cv2.imwrite(save_path, self.enhanced_image):
            self.status_label.setText(f"Imej disimpan ke {save_path}.")
        else:
            self.status_label.setText("Gagal menyimpan imej.")

    def save_metrics_to_excel(self, file_path, metrics):
        self.metrics_data.append(
            {
                "Image Path": file_path,
                "Entropi Input": metrics[0],
                "Entropi Output": metrics[1],
                "CI Input": metrics[2],
                "CI Output": metrics[3],
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        pd.DataFrame(self.metrics_data).to_excel("test.xlsx", index=False)

    def convert_cv_to_pixmap(self, cv_img):
        height, width, _ = cv_img.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            cv_img.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).rgbSwapped()
        return QPixmap.fromImage(q_image.copy())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageEnhancerApp()
    window.show()
    sys.exit(app.exec_())
