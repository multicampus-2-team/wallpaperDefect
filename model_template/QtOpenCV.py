import cv2 as cv
import sys
import os
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

class ShowVideo(QtCore.QObject):

    flag = 0
    camera = cv.VideoCapture(0)

    ret, image = camera.read()
    height, width = image.shape[:2]

    VideoSignal1 = QtCore.pyqtSignal(QtGui.QImage)
    VideoSignal2 = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)
        #self.setupUi(self)
        #self.fontSize = 10

        #self.btn_printIdentifyText.clicked.connect(self.printTextEdit)

        #self.textEdit = QtWidgets.QTextEdit()
        #layout = QtWidgets.QVBowLayout()
        #layout.addWidget(self.textEdit)
        #self.setLayout(layout)

    def printTextEdit(self):
        print(self.textedit_Test.toPlainText())

    @QtCore.pyqtSlot()
    def startVideo(self):
        global image

        run_video = True
        while run_video:
            ret, image = self.camera.read()
            color_swapped_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            qt_image1 = QtGui.QImage(color_swapped_image.data,
                                    self.width,
                                    self.height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)
            self.VideoSignal1.emit(qt_image1)


            if self.flag:
                img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                img_canny = cv.Canny(img_gray, 50, 100)
                img_dilated = cv.dilate(img_canny, (7, 7), iterations=3)
                img_eroded = cv.erode(img_dilated, (7, 7), iterations=3)

                qt_image2 = QtGui.QImage(img_canny.data,
                                         self.width,
                                         self.height,
                                         img_canny.strides[0],
                                         QtGui.QImage.Format_Grayscale8)

                self.VideoSignal2.emit(qt_image2)



            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(25, loop.quit) #25 ms
            loop.exec_()

    @QtCore.pyqtSlot()
    def canny(self):
        self.flag = 1 - self.flag

    def identify(self):
        #cap = cv.VideoCapture(0)
        #cap_canny = cv.Canny(cap, 50, 100)
        #cap_dilated = cv.dilate(cap_canny, (7, 7), iterations=3)
        #cap_eroded = cv.erode(cap_dilated, (7, 7), iterations=3)

        #cv.imwrite('../../open/Codes/basic/cap_eroded.png', cap_eroded)
        self.ident = os.system("python basic_CNN_2class.py")

        #text = open('identified_output.txt').read()
        #QtWidgets.QLineEdit.information(self,"info",text)
        #text_edit.clear()
        #try:
        #    file_name = open('identified_output.txt').read()
        #    File = open(file_name[0], "r", encoding='utf-8')
        #    lines = File.readlines()
        #except:
        #    pass



class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        self.setWindowTitle('Test')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    thread = QtCore.QThread()
    thread.start()
    vid = ShowVideo()
    vid.moveToThread(thread)

    image_viewer1 = ImageViewer()
    image_viewer2 = ImageViewer()

    vid.VideoSignal1.connect(image_viewer1.setImage)
    vid.VideoSignal2.connect(image_viewer2.setImage)

    push_button1 = QtWidgets.QPushButton('Camera')
    push_button2 = QtWidgets.QPushButton('Image Processing')
    push_button3 = QtWidgets.QPushButton('Identify')
    push_button1.clicked.connect(vid.startVideo)
    push_button2.clicked.connect(vid.canny)
    push_button3.clicked.connect(vid.identify)

    vertical_layout = QtWidgets.QVBoxLayout()
    horizontal_layout = QtWidgets.QHBoxLayout()
    horizontal_layout.addWidget(image_viewer1)
    horizontal_layout.addWidget(image_viewer2)
    vertical_layout.addLayout(horizontal_layout)
    vertical_layout.addWidget(push_button1)
    vertical_layout.addWidget(push_button2)
    vertical_layout.addWidget(push_button3)

    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)

    text_edit = QtWidgets.QLineEdit("텍스트")
    vertical_layout.addWidget(text_edit)

    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    sys.exit(app.exec_())
