import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QFileDialog, QWidget, QMenuBar, QAction, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from datetime import datetime
import traceback

# Define a function to compute the dark channel of the image
def DarkChannel(im, sz):
    b, g, r = cv2.split(im)  # Split image into B, G, R channels
    dc = cv2.min(cv2.min(r, g), b)  # Find the minimum across all channels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))  # Create a morphological structuring element
    dark = cv2.erode(dc, kernel)  # Apply erosion to get the dark channel
    return dark

# Function to estimate atmospheric light from the dark channel
def AtmLight(im, dark):
    [h, w] = im.shape[:2]  # Get image dimensions
    imsz = h * w  # Calculate total number of pixels
    numpx = int(max(math.floor(imsz / 1000), 1))  # Select the top 0.1% brightest pixels
    darkvec = dark.reshape(imsz)  # Flatten dark channel to a 1D array
    imvec = im.reshape(imsz, 3)  # Flatten image to a 2D array (pixels x 3 channels)
    indices = darkvec.argsort()  # Sort dark channel pixel intensities
    indices = indices[imsz - numpx::]  # Select indices of the brightest pixels
    atmsum = np.zeros([1, 3])  # Initialize the atmospheric light sum
    for ind in range(1, numpx):  # Sum the RGB values of the brightest pixels
        atmsum = atmsum + imvec[indices[ind]]
    A = atmsum / numpx  # Compute average atmospheric light
    return A / np.max(A)  # Normalize atmospheric light

# Estimate the transmission map using the atmospheric light and dark channel
def TransmissionEstimate(im, A, sz):
    omega = 0.95  # Parameter to control the amount of haze removal
    im3 = np.empty(im.shape, im.dtype)  # Create an empty array of the same shape as input image
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]  # Normalize image channels by atmospheric light
    transmission = 1 - omega * DarkChannel(im3, sz)  # Estimate transmission
    return transmission

# Perform guided filtering to refine the transmission map
def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))  # Compute mean of input image
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))  # Compute mean of input guidance image
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))  # Compute mean of input image x guidance image
    cov_Ip = mean_Ip - mean_I * mean_p  # Compute covariance of input and guidance image
    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))  # Compute mean of input image squared
    var_I = mean_II - mean_I * mean_I  # Compute variance of input image
    a = cov_Ip / (var_I + eps)  # Calculate slope of the linear model
    b = mean_p - a * mean_I  # Calculate intercept of the linear model
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))  # Smooth slope
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))  # Smooth intercept
    q = mean_a * im + mean_b  # Compute the refined output
    return q

# Refine the transmission map using guided filtering
def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    gray = np.float64(gray) / 255  # Normalize grayscale values to [0, 1]
    r = 60  # Radius for guided filter
    eps = 0.0001  # Regularization parameter for guided filter
    t = Guidedfilter(gray, et, r, eps)  # Apply guided filter
    return t

# Recover the final dehazed image
def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)  # Create an empty array for the result
    t = cv2.max(t, tx)  # Avoid division by zero by setting a lower limit for transmission
    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]  # Recover image for each channel
    res = np.clip(res, 0, 1)  # Clip pixel values to [0, 1]
    return res

def enhance_image(file_path, target_size=None):
    """
    Enhance an overexposed BGR image.

    Parameters
    ----------
    file_path : str
        Input image path.

    target_size : tuple or None
        Maximum (width, height). Use None to retain full resolution.
        The GUI should resize only the displayed QPixmap.

    Returns
    -------
    original_image, enhanced_image : numpy.ndarray
        Original/working image and enhanced BGR image.
    """

    src = cv2.imread(file_path, cv2.IMREAD_COLOR)

    if src is None:
        raise FileNotFoundError(
            f"Image file '{file_path}' could not be loaded."
        )

    # ---------------------------------------------------------
    # 1. Preserve maximum available resolution
    # ---------------------------------------------------------
    if target_size is not None:
        height, width = src.shape[:2]

        scale = min(
            target_size[0] / width,
            target_size[1] / height,
            1.0
        )

        if scale < 1.0:
            new_size = (
                max(1, round(width * scale)),
                max(1, round(height * scale))
            )

            working_image = cv2.resize(
                src,
                new_size,
                interpolation=cv2.INTER_AREA
            )
        else:
            working_image = src.copy()
    else:
        working_image = src.copy()

    # Convert to floating point.
    image_float = (
        working_image.astype(np.float32) / 255.0
    )

    # ---------------------------------------------------------
    # 2. Detect partially and completely clipped highlights
    # ---------------------------------------------------------
    maximum_channel = np.max(image_float, axis=2)
    minimum_channel = np.min(image_float, axis=2)

    # Partially overexposed: at least one channel is near clipping.
    highlight_mask = np.clip(
        (maximum_channel - 0.72) / 0.28,
        0.0,
        1.0
    )

    # Smoothstep produces gradual transitions.
    highlight_mask = (
        highlight_mask
        * highlight_mask
        * (3.0 - 2.0 * highlight_mask)
    )

    highlight_mask = cv2.GaussianBlur(
        highlight_mask,
        (0, 0),
        sigmaX=3.0
    )

    # Fully clipped pixels contain almost no useful channel data.
    fully_clipped_mask = np.uint8(
        (minimum_channel >= 0.985) * 255
    )

    # Remove isolated clipping-mask noise.
    clipping_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    fully_clipped_mask = cv2.morphologyEx(
        fully_clipped_mask,
        cv2.MORPH_OPEN,
        clipping_kernel
    )

    fully_clipped_mask = cv2.dilate(
        fully_clipped_mask,
        clipping_kernel,
        iterations=1
    )

    # ---------------------------------------------------------
    # 3. Gentle denoising
    # ---------------------------------------------------------
    denoised = cv2.fastNlMeansDenoisingColored(
        working_image,
        None,
        h=2,
        hColor=3,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # ---------------------------------------------------------
    # 4. Separate luminance from colour
    # ---------------------------------------------------------
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)

    luminance_float = (
        luminance.astype(np.float32) / 255.0
    )

    # ---------------------------------------------------------
    # 5. Correct overall exposure using image statistics
    # ---------------------------------------------------------
    percentile_95 = float(
        np.percentile(luminance_float, 95)
    )

    if percentile_95 > 1e-6:
        exposure_scale = float(
            np.clip(
                0.88 / percentile_95,
                0.68,
                1.0
            )
        )
    else:
        exposure_scale = 1.0

    exposed_luminance = np.clip(
        luminance_float * exposure_scale,
        0.0,
        1.0
    )

    # ---------------------------------------------------------
    # 6. Highlight compression
    # ---------------------------------------------------------
    # This curve compresses highlights without flattening shadows.
    compression_strength = 2.2

    compressed_luminance = (
        exposed_luminance
        / (
            exposed_luminance
            + compression_strength
            * (1.0 - exposed_luminance)
            + 1e-6
        )
    )

    # Normalize the tone curve.
    white_value = 1.0 / (
        1.0
        + compression_strength * (1.0 - 1.0)
    )

    compressed_luminance /= white_value

    # Apply compression mainly to highlights.
    recovered_luminance = (
        exposed_luminance * (1.0 - highlight_mask)
        + compressed_luminance * highlight_mask
    )

    # Gently darken bright midtones.
    recovered_luminance = np.power(
        np.clip(recovered_luminance, 0.0, 1.0),
        1.06
    )

    recovered_luminance_u8 = np.uint8(
        np.clip(
            recovered_luminance * 255.0,
            0,
            255
        )
    )

    # ---------------------------------------------------------
    # 7. Recover local contrast
    # ---------------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    clahe_luminance = clahe.apply(
        recovered_luminance_u8
    )

    # Reduce CLAHE strength in the brightest regions to avoid noise.
    clahe_strength = (
        0.60 - 0.35 * highlight_mask
    ).astype(np.float32)

    local_contrast = (
        recovered_luminance_u8.astype(np.float32)
        * (1.0 - clahe_strength)
        + clahe_luminance.astype(np.float32)
        * clahe_strength
    )

    # ---------------------------------------------------------
    # 8. Multi-scale detail enhancement
    # ---------------------------------------------------------
    fine_base = cv2.GaussianBlur(
        local_contrast,
        (0, 0),
        sigmaX=0.8
    )

    medium_base = cv2.GaussianBlur(
        local_contrast,
        (0, 0),
        sigmaX=2.0
    )

    large_base = cv2.bilateralFilter(
        np.uint8(np.clip(local_contrast, 0, 255)),
        d=9,
        sigmaColor=25,
        sigmaSpace=25
    ).astype(np.float32)

    fine_detail = local_contrast - fine_base
    medium_detail = fine_base - medium_base
    structural_detail = medium_base - large_base

    # Symmetrical clipping preserves positive and negative edges.
    fine_detail = np.clip(fine_detail, -6.0, 6.0)
    medium_detail = np.clip(medium_detail, -10.0, 10.0)
    structural_detail = np.clip(
        structural_detail,
        -14.0,
        14.0
    )

    # Do not strongly sharpen unreliable clipped pixels.
    reliable_detail_mask = (
        1.0 - 0.80 * highlight_mask
    )

    clipped_float = (
        fully_clipped_mask.astype(np.float32) / 255.0
    )

    reliable_detail_mask *= (
        1.0 - 0.90 * clipped_float
    )

    detailed_luminance = (
        large_base
        + 1.05 * structural_detail * reliable_detail_mask
        + 1.10 * medium_detail * reliable_detail_mask
        + 0.85 * fine_detail * reliable_detail_mask
    )

    detailed_luminance = np.uint8(
        np.clip(detailed_luminance, 0, 255)
    )

    # ---------------------------------------------------------
    # 9. Reconstruct plausible colour in fully clipped areas
    # ---------------------------------------------------------
    if np.any(fully_clipped_mask):
        channel_a = cv2.inpaint(
            channel_a,
            fully_clipped_mask,
            3,
            cv2.INPAINT_TELEA
        )

        channel_b = cv2.inpaint(
            channel_b,
            fully_clipped_mask,
            3,
            cv2.INPAINT_TELEA
        )

    enhanced_lab = cv2.merge(
        (
            detailed_luminance,
            channel_a,
            channel_b
        )
    )

    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    # ---------------------------------------------------------
    # 10. Natural colour and final edge-preserving cleanup
    # ---------------------------------------------------------
    hsv = cv2.cvtColor(
        enhanced_image,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    # Mild saturation improvement only.
    hsv[:, :, 1] *= 1.04
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1],
        0,
        255
    )

    enhanced_image = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR
    )

    enhanced_image = cv2.bilateralFilter(
        enhanced_image,
        d=5,
        sigmaColor=10,
        sigmaSpace=10
    )

    return working_image, enhanced_image


# Function to calculate image entropy
def calculate_entropy(image):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256), density=True)  # Compute histogram
    histogram = histogram[histogram > 0]  # Remove zero values
    entropy = -np.sum(histogram * np.log2(histogram))  # Compute entropy
    return entropy

# Function to calculate colorfulness index
def calculate_colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))  # Split into color channels
    rg = R - G  # Compute red-green difference
    yb = 0.5 * (R + G) - B  # Compute yellow-blue difference
    std_rg = np.std(rg)  # Standard deviation of rg
    std_yb = np.std(yb)  # Standard deviation of yb
    mean_rg = np.mean(rg)  # Mean of rg
    mean_yb = np.mean(yb)  # Mean of yb
    colorfulness = np.sqrt((std_rg ** 2) + (std_yb ** 2)) + 0.3 * np.sqrt((mean_rg ** 2) + (mean_yb ** 2))  # Compute colorfulness
    return colorfulness

def calculate_metrics(original, enhanced):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    entropy_original = calculate_entropy(original_gray)  # Compute entropy for original
    entropy_enhanced = calculate_entropy(enhanced_gray)  # Compute entropy for enhanced
    
    colorfulness_original = calculate_colorfulness(original)  # Compute colorfulness for original
    colorfulness_enhanced = calculate_colorfulness(enhanced)  # Compute colorfulness for enhanced

    return entropy_original, entropy_enhanced, colorfulness_original, colorfulness_enhanced

class ImageEnhancerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.metrics_data = []  # Initialize metrics_data as an empty list
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Sistem Pemulihan Imej")
        self.setGeometry(100, 100, 1000, 800)

        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("background-color: grey;")
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.header_layout = QHBoxLayout()
        self.image_layout = QGridLayout()  # Changed to QGridLayout for flexibility
        self.status_layout = QVBoxLayout()

        self.custom_text = QLabel("Overexposed Image Enhancement System")
        self.custom_text.setAlignment(Qt.AlignCenter)
        self.custom_text.setStyleSheet("font-size: 30px; font-weight: bold;")
        self.header_layout.addWidget(self.custom_text)

        # Image display area
        self.original_label = QLabel("Imej Input")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 3px solid black; color: black;")
        self.original_label.setFixedSize(500, 380)

        self.enhanced_label = QLabel("Imej Dipuihkan Dengan Teknik Integrasi 3 Teknik")
        self.enhanced_label.setAlignment(Qt.AlignCenter)
        self.enhanced_label.setStyleSheet("border: 3px solid green; color: black;")
        self.enhanced_label.setFixedSize(500, 380)

        self.original_graph_canvas = FigureCanvas(plt.figure(figsize=(3, 2)))
        self.enhanced_graph_canvas = FigureCanvas(plt.figure(figsize=(3, 2)))
        
        # Set the fixed size for all canvases (in pixels)
        self.original_graph_canvas.setFixedSize(500, 380)
        self.enhanced_graph_canvas.setFixedSize(500, 380)

        # Arrange the image and graph canvases in a grid (rows, columns)
        self.image_layout.addWidget(self.original_label, 0, 0)
        self.image_layout.addWidget(self.original_graph_canvas, 0, 2)
        self.image_layout.addWidget(self.enhanced_label, 1, 0)
        self.image_layout.addWidget(self.enhanced_graph_canvas, 1, 2)

        # Status label
        self.status_label = QLabel("Muat naik Imej untuk memaparkan metrik")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 15px; background-color: black; color: white;")

        self.main_layout.addLayout(self.header_layout)
        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addWidget(self.status_label)

        # Menu bar
        menu_bar = QMenuBar(self)
        file_menu = menu_bar.addMenu("Fail")

        load_action = QAction("Muat naik Imej", self)
        load_action.triggered.connect(self.load_images)

        #load_groundtruth_action = QAction("Load Ground Truth", self)
        #load_groundtruth_action.triggered.connect(self.load_groundtruth)

        save_action = QAction("Simpan Imej Pulih", self)
        save_action.triggered.connect(self.save_image)

        file_menu.addAction(load_action)
        #file_menu.addAction(load_groundtruth_action)
        file_menu.addAction(save_action)

    def plot_noise_graph(self, original_image, enhanced_image):
        # Convert images to numpy arrays
        original_image_np = np.array(original_image, dtype=np.float32)
        enhanced_image_np = np.array(enhanced_image, dtype=np.float32)

        # Compute noise (difference between original and enhanced images)
        noise_image_np = original_image_np - enhanced_image_np

        # Colors for RGB channels
        colors = ('r', 'g', 'b')

        # Clear previous plots
        self.original_graph_canvas.figure.clear()
        self.enhanced_graph_canvas.figure.clear()

        # Create new subplots
        ax1 = self.original_graph_canvas.figure.add_subplot(111)
        ax2 = self.enhanced_graph_canvas.figure.add_subplot(111)

        # Plot noise frequency distribution
        for i, color in enumerate(colors):
            noise_values = original_image_np[..., i].ravel()
            hist, bins = np.histogram(noise_values, bins=512, range=(-255, 255), density=True)
            hist_smoothed = gaussian_filter1d(hist, sigma=2)
            ax1.plot(bins[:-1], hist_smoothed, color=color, label=f'{color.upper()} Channel', linewidth=1.5, alpha=0.8)
        
        ax1.set_title('Frekuensi Hingar Imej Asal')
        ax1.set_xlabel('Kekerapan Hingar')
        ax1.set_ylabel('Frekuensi')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Plot noise frequency distribution for enhanced image (if needed)
        for i, color in enumerate(colors):
            noise_values = enhanced_image_np[..., i].ravel()
            hist, bins = np.histogram(noise_values, bins=512, range=(-255, 255), density=True)
            hist_smoothed = gaussian_filter1d(hist, sigma=2)
            ax2.plot(bins[:-1], hist_smoothed, color=color, label=f'{color.upper()} Channel', linewidth=1.5, alpha=0.8)
        
        ax2.set_title('Frekuensi Hingar Imej Pulih')
        ax2.set_xlabel('Kekerapan Hingar')
        ax2.set_ylabel('Frekuensi')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Refresh the canvases
        self.original_graph_canvas.draw()
        self.enhanced_graph_canvas.draw()

    def load_images(self):
        self.original_graph_canvas.figure.clear()
        self.enhanced_graph_canvas.figure.clear()
        options = QFileDialog.Options()
        
        file_path_1, _ = QFileDialog.getOpenFileName(self, "Buka Imej Input", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        
        if file_path_1:
            self.display_images(file_path_1)
        else:
            print("Sila pilih 3 file yang sesuai.")
            
    def save_image(self):
        if self.enhanced_label.pixmap() is not None:
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Simpan Imej", "", "Images (*.png *.jpg *.jpeg *.bmp)"
            )
            if save_path:
                # Capture the pixmap displayed in the enhanced_label
                pixmap = self.enhanced_label.pixmap()
                # Save the pixmap to the chosen file path
                if pixmap.save(save_path):
                    self.status_label.setText(f"Imej disimpan ke {save_path}.")
                else:
                    self.status_label.setText("Gagal menyimpan imej.")
        else:
            self.status_label.setText("Tiada imej pulih untuk disimpan.")
            
    def save_metrics_to_excel(self, file_path, metrics):
        # Extract metrics
        entropy_original, entropy_enhanced, colorfulness_original, colorfulness_enhanced = metrics

        # Create a dictionary for the current metrics
        metric_entry = {
            "Image Path": file_path,
            "Entropi Asal": entropy_original,
            "Entropi Pulih": entropy_enhanced,
            "CI Asal": colorfulness_original,
            "CI Pulih": colorfulness_enhanced,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Append the metrics to the list
        self.metrics_data.append(metric_entry)

        # Convert the list to a DataFrame
        df = pd.DataFrame(self.metrics_data)

        # Save the DataFrame to an Excel file
        excel_file = "data.xlsx"
        df.to_excel(excel_file, index=False)
    
    def display_images(self, file_path):
        try:
            original_image, enhanced_image = enhance_image(file_path)

            # Convert OpenCV images to QPixmap
            original_qpixmap = self.convert_cv_to_pixmap(original_image)
            enhanced_qpixmap = self.convert_cv_to_pixmap(enhanced_image)

            # Scale images to fit their prepared label frames
            original_qpixmap = original_qpixmap.scaled(
                self.original_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            enhanced_qpixmap = enhanced_qpixmap.scaled(
                self.enhanced_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Display images
            self.original_label.setPixmap(original_qpixmap)
            self.original_label.setToolTip(f"Imej Input: {file_path}")

            self.enhanced_label.setPixmap(enhanced_qpixmap)
            self.enhanced_label.setToolTip(f"Imej Pulih: {file_path}")

            # Keep the full processed image for saving
            self.enhanced_image = enhanced_image.copy()

            metrics = calculate_metrics(
                original_image,
                enhanced_image
            )

            self.status_label.setText(
                f"Entropi Asal: {metrics[0]:.4f}, "
                f"Entropi Pulih: {metrics[1]:.4f}   "
                f"CI Asal: {metrics[2]:.4f}, "
                f"CI Pulih: {metrics[3]:.4f}"
            )

            self.save_metrics_to_excel(file_path, metrics)
            self.plot_noise_graph(original_image, enhanced_image)

        except Exception as e:
            self.status_label.setText(f"Ralat: {str(e)}")
            print(f"Error in display_images: {e}")
        
    def convert_cv_to_pixmap(self, cv_img):
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        return QPixmap.fromImage(q_image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageEnhancerApp()
    window.show()
    sys.exit(app.exec_())
