import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QDialog, QVBoxLayout, QLabel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 400, 300)
        
        self.button = QPushButton("Open Popup", self)
        self.button.setGeometry(150, 130, 100, 40)
        self.button.clicked.connect(self.open_popup)

    def open_popup(self):
        self.popup = PopupWindow(self)
        self.popup.exec()

class PopupWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Popup Window")
        self.setGeometry(150, 150, 200, 100)
        
        layout = QVBoxLayout()
        label = QLabel("This is a popup window", self)
        layout.addWidget(label)
        
        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())