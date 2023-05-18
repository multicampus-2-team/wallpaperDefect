import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_main = uic.loadUiType("firstQt.ui")[0]  # ui 파일 불러오기


class MainWindow(QMainWindow, QWidget, form_main):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.show()

    def initUI(self):
        self.setupUi(self)
        self.pushButton.clicked.connect(self.buttonClicked)

    def buttonClicked(self):
        self.hide()
        self.second = os.system("python SecondQt.py")
        self.second.exec()
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    sys.exit(app.exec_())
