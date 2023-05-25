import sys
import cv2
import os
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget, QMessageBox, QFileDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("도배 하자 유형 분류기")
        self.setGeometry(100, 100, 800, 600)

        self.camera_label = QLabel(self)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)  # 최소 크기 설정

        self.text_edit = QTextEdit(self)
        self.setCentralWidget(self.text_edit)

        self.camera_button = QPushButton("카메라 열기", self)
        self.camera_button.clicked.connect(self.start_camera)

        self.capture_button = QPushButton("캡처", self)
        self.capture_button.clicked.connect(self.capture_frame)

        self.prediction_button = QPushButton("유형 예측", self)
        self.prediction_button.clicked.connect(self.predict)

        layout = QVBoxLayout()
        layout.addWidget(self.camera_label)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.camera_button)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.prediction_button)

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.camera = cv2.VideoCapture(0)
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.display_camera_frame)

    def start_camera(self):
        if not self.camera_timer.isActive():
            self.camera_button.setText("카메라 끄기")
            self.camera_timer.start(30)
        else:
            self.camera_button.setText("카메라 열기")
            self.camera_timer.stop()
            self.camera.release()
            self.camera_label.clear()

    def display_camera_frame(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio))

    def capture_frame(self):
        ret, frame = self.camera.read()
        if ret:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG Files (*.jpg)")
            if file_path:
                cv2.imwrite(file_path, frame)
                QMessageBox.information(self, "Capture", "Image captured successfully!")

    def predict(self):
        #cap = cv.VideoCapture(0)
        #cap_canny = cv.Canny(cap, 50, 100)
        #cap_dilated = cv.dilate(cap_canny, (7, 7), iterations=3)
        #cap_eroded = cv.erode(cap_dilated, (7, 7), iterations=3)

        #cv.imwrite('../../open/Codes/basic/cap_eroded.png', cap_eroded)
        self.ident = os.system("python PyTorchMayProject.py")

        file_path='prediction.txt'

        try:
            with open(file_path, 'r') as file:
                text = file.read()
                self.text_edit.setPlainText(text)
        except Exception as e:
            print(f"Failed to load text file: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
