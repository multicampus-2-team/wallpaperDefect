import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow

form_secondwindow = uic.loadUiType("secondQt.ui")[0]  # 두 번째창 ui


class SecondWindow(QMainWindow, QWidget, form_secondwindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.show()

    def initUI(self):
        self.setupUi(self)
        self.pushButton.clicked.connect(self.buttonClicked)

    def buttonClicked(self):
        self.hide()
        self.third = os.system("python QtOpenCV.py")
        self.third.exec()
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SecondWindow()
    sys.exit(app.exec_())