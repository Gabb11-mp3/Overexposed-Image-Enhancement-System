import sys
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
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

# Enhance the image by applying dehazing, detail enhancement, and filtering
def enhance_image(file_path, target_size=(500, 500)):
    src = cv2.imread(file_path)  # Load the input image
    if src is None:
        raise FileNotFoundError(f"Image file '{file_path}' not found or could not be loaded.")
    h, w = src.shape[:2]  # Get original dimensions
    scale_factor = min(target_size[1] / h, target_size[0] / w)  # Calculate scaling factor
    new_size = (int(w * scale_factor), int(h * scale_factor))  # Compute new dimensions
    resized_src = cv2.resize(src, new_size, interpolation=cv2.INTER_AREA)  # Resize the image
    I = resized_src.astype('float64') / 255  # Normalize image to [0, 1]

    # Dehazing process
    dark = DarkChannel(I, 15)  # Compute dark channel
    A = AtmLight(I, dark)  # Estimate atmospheric light
    te = TransmissionEstimate(I, A, 15)  # Estimate transmission map
    t = TransmissionRefine(resized_src, te)  # Refine transmission map
    J = Recover(I, t, A, 0.1)  # Recover dehazed image
    J = (J * 255).astype('uint8')  # Convert to 8-bit image

    # Apply additional enhancements
    filtered_image = cv2.bilateralFilter(J, d=2, sigmaColor=80, sigmaSpace=80)  # Bilateral filter
    denoised_image = cv2.fastNlMeansDenoisingColored(filtered_image, None, 3, 3, 7, 15)  # Denoising
    blended_image = cv2.addWeighted(filtered_image, 0.9, denoised_image, 0.1, 0)  # Blending

    # Extract fine details
    detail_layer = cv2.subtract(filtered_image, cv2.GaussianBlur(filtered_image, (5, 5), 2.0))  # Fine details
    detail_layer = cv2.addWeighted(detail_layer, 1, blended_image, 1, 0)  # Blend with filtered image

    # Final sharpening
    gaussian_blurred = cv2.GaussianBlur(detail_layer, (7, 7), 1.5)  # Gaussian blur for unsharp masking
    enhanced_image = cv2.addWeighted(detail_layer, 1.5, gaussian_blurred, -0.5, 0)  # Unsharp masking

    return resized_src, enhanced_image  # Return original and enhanced images

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

def calculate_metrics(ground_truth, original, enhanced):
    # Convert images to grayscale for PSNR and SSIM evaluation
    ground_truth_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # Calculate PSNR  for output image compared to ground truth
    psnr = compare_psnr(ground_truth_gray, enhanced_gray)

    # Calculate SSIM for output image compared to ground truth
    ssim = compare_ssim(ground_truth_gray, enhanced_gray)
    
    entropy_original = calculate_entropy(original_gray)  # Compute entropy for original
    entropy_enhanced = calculate_entropy(enhanced_gray)  # Compute entropy for enhanced
    
    colorfulness_original = calculate_colorfulness(original)  # Compute colorfulness for original
    colorfulness_enhanced = calculate_colorfulness(enhanced)  # Compute colorfulness for enhanced

    return psnr, ssim, entropy_original, entropy_enhanced, colorfulness_original, colorfulness_enhanced

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

        # Add logos and text to the header
        self.logo_left = QLabel()
        self.logo_left.setPixmap(QPixmap("C:/fypGibo/Logo UMS putih.png").scaled(400, 400, Qt.KeepAspectRatio))
        self.logo_left.setAlignment(Qt.AlignCenter)

        self.logo_right = QLabel()
        self.logo_right.setPixmap(QPixmap("C:/fypGibo/logo mcg.png").scaled(400, 400, Qt.KeepAspectRatio))
        self.logo_right.setAlignment(Qt.AlignCenter)

        self.custom_text = QLabel("""DISEDIAKAN OLEH: GABRIEL DENNIS
    DIPANTAU OLEH: PROF DR. ABDUULLAH BADE
    TAJUK KAJIAN: Penambahbaikkan Kualiti Imej Berkeamatan 
    Tinggi Tunggal Menggunakan DARK CHANNEL PRIOR(DCP)
                                    """)
        self.custom_text.setAlignment(Qt.AlignCenter)
        self.custom_text.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.header_layout.addWidget(self.logo_left)
        self.header_layout.addWidget(self.custom_text)
        self.header_layout.addWidget(self.logo_right)

        # Image display area
        self.original_label = QLabel("Imej Input")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 3px solid black; color: black;")
        self.original_label.setFixedSize(500, 380)

        self.enhanced_label = QLabel("Imej Dipuihkan Dengan Teknik Integrasi 3 Teknik")
        self.enhanced_label.setAlignment(Qt.AlignCenter)
        self.enhanced_label.setStyleSheet("border: 3px solid green; color: black;")
        self.enhanced_label.setFixedSize(500, 380)

        self.groundtruth_label = QLabel("Imej Rujukan")
        self.groundtruth_label.setAlignment(Qt.AlignCenter)
        self.groundtruth_label.setStyleSheet("border: 3px solid white; color: black;")
        self.groundtruth_label.setFixedSize(500, 380)
        
        self.base_label = QLabel("Imej Dipulihkan Dengan Teknik DCP Asal")
        self.base_label.setAlignment(Qt.AlignCenter)
        self.base_label.setStyleSheet("border: 3px solid red; color: black;")
        self.base_label.setFixedSize(500, 380)

        self.original_graph_canvas = FigureCanvas(plt.figure(figsize=(3, 2)))
        self.enhanced_graph_canvas = FigureCanvas(plt.figure(figsize=(3, 2)))
        
        # Set the fixed size for all canvases (in pixels)
        self.original_graph_canvas.setFixedSize(500, 380)
        self.enhanced_graph_canvas.setFixedSize(500, 380)

        # Arrange the image and graph canvases in a grid (rows, columns)
        self.image_layout.addWidget(self.original_label, 0, 0)
        self.image_layout.addWidget(self.original_graph_canvas, 0, 2)
        self.image_layout.addWidget(self.enhanced_label, 0, 1)
        self.image_layout.addWidget(self.enhanced_graph_canvas, 1, 2)
        self.image_layout.addWidget(self.groundtruth_label, 1, 0)
        self.image_layout.addWidget(self.base_label, 1, 1)

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
        file_path_2, _ = QFileDialog.getOpenFileName(self, "Buka Imej Rujukan", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        file_path_3, _ = QFileDialog.getOpenFileName(self, "Buka Imej Hasil DCP Asal", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        
        if file_path_1 and file_path_2:
            self.display_images(file_path_1)
            self.display_groundtruth_image(file_path_2)
            self.display_base(file_path_3)
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
        psnr, ssim, entropy_original, entropy_enhanced, colorfulness_original, colorfulness_enhanced = metrics

        # Create a dictionary for the current metrics
        metric_entry = {
            "Image Path": file_path,
            "PSNR": psnr,
            "SSIM": ssim,
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
        excel_file = "test.xlsx"
        df.to_excel(excel_file, index=False)
    
    def display_images(self, file_path):
        try:
            # Enhance image (this function should return original and enhanced images)
            original_image, enhanced_image = enhance_image(file_path)
            
            # Display original and enhanced images
            original_qpixmap = self.convert_cv_to_pixmap(original_image)
            self.original_label.setPixmap(original_qpixmap)
            self.original_label.setToolTip(f"Imej Input: {file_path}")  # Add file name as tooltip
            
            enhanced_qpixmap = self.convert_cv_to_pixmap(enhanced_image)
            self.enhanced_label.setPixmap(enhanced_qpixmap)
            self.enhanced_label.setToolTip(f"Imej Pulih: {file_path}")  # Add file name as tooltip
            
            # Load the ground truth image (assuming it's in the same directory with a specific naming convention)
            ground_truth_path = file_path.replace(".jpg", "_gt.jpg")  # Adjust this based on your ground truth file naming
            ground_truth_image = cv2.imread(ground_truth_path)
            
            if ground_truth_image is None:
                raise FileNotFoundError(f"Imej rujukan '{ground_truth_path}' tidak dijumpai aau dimuat naik")
            
            # Resize the ground truth image to match the original image size
            ground_truth_image = cv2.resize(ground_truth_image, (original_image.shape[1], original_image.shape[0]))
            
            # Display the ground truth image
            ground_truth_qpixmap = self.convert_cv_to_pixmap(ground_truth_image)
            self.groundtruth_label.setPixmap(ground_truth_qpixmap)
            self.groundtruth_label.setToolTip(f"Imej Rujukan: {ground_truth_path}")  # Add file name as tooltip
            
            # Calculate and display metrics
            metrics = calculate_metrics(
                ground_truth_image, original_image, enhanced_image
            )
            
            # Update the status label with all metrics
            self.status_label.setText(
                f"PSNR: {metrics[0]:.4f}   "
                f"SSIM: {metrics[1]:.4f}   "
                f"Entrofi Asal: {metrics[2]:.4f}, Entrofi Pulih: {metrics[3]:.4f}   "
                f"CI Asal: {metrics[4]:.4f}, CI Pulih: {metrics[5]:.4f}   "
            )

            # Save metrics to Excel
            self.save_metrics_to_excel(file_path, metrics)
        except Exception as e:
            self.status_label.setText(f"Ralat: {str(e)}")
        
        # Call plot_image_graph function to display the RGB histograms
        self.plot_noise_graph(original_image, enhanced_image)

    def display_groundtruth_image(self, file_path_2, target_size=(500, 500)):
        try:
            # Load ground truth image using OpenCV
            groundtruth = cv2.imread(file_path_2)
            if groundtruth is None:
                raise FileNotFoundError(f"Fail imej '{file_path_2}' tidak dijumpai atau tidak dimuat naik")

            # Resize the image while maintaining aspect ratio
            h, w = groundtruth.shape[:2]
            scale_factor = min(target_size[1] / h, target_size[0] / w)
            new_size = (int(w * scale_factor), int(h * scale_factor))
            groundtruth_image = cv2.resize(groundtruth, new_size, interpolation=cv2.INTER_AREA)

            # Convert image to QPixmap
            groundtruth_qpixmap = self.convert_cv_to_pixmap(groundtruth_image)
            self.groundtruth_label.setPixmap(groundtruth_qpixmap)
            self.groundtruth_label.setToolTip(f"Imej Rujukan: {file_path_2}")  # Add file name as tooltip

            # Force UI update
            self.groundtruth_label.repaint()

            # Display file path as label
            self.groundtruth_label.setToolTip(f"Imej Rujukan: {file_path_2}")
        except Exception as e:
            self.status_label.setText(f"Ralat: {str(e)}")
            
    def display_base(self, file_path_3, target_size=(500, 500)):
        try:
            # Load base DCP image using OpenCV
            base = cv2.imread(file_path_3)
            if base is None:
                raise FileNotFoundError(f"Fail imej '{file_path_3}' tidak dijumpai atau dimuat naik.")

            # Resize the image while maintaining aspect ratio
            h, w = base.shape[:2]
            scale_factor = min(target_size[1] / h, target_size[0] / w)
            new_size = (int(w * scale_factor), int(h * scale_factor))
            base_image = cv2.resize(base, new_size, interpolation=cv2.INTER_AREA)

            # Convert image to QPixmap
            base_qpixmap = self.convert_cv_to_pixmap(base_image)
            self.base_label.setPixmap(base_qpixmap)
            self.base_label.setToolTip(f"DCP Asal: {file_path_3}")  # Add file name as tooltip

            # Force UI update
            self.base_label.repaint()

            # Display file path as label
            self.base_label.setToolTip(f"DCP Asal: {file_path_3}")
        except Exception as e:
            self.status_label.setText(f"Ralat: {str(e)}")
        
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
