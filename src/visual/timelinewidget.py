import os.path
from PyQt4.QtCore import *
from PyQt4.QtGui import *

def find(filename):
    if __file__ is None:
        return filename
    else:
        return os.path.join(os.path.dirname(__file__), filename)

class TimeLineWidget(QWidget):
    togglePlay = pyqtSignal()
    rewind = pyqtSignal()
    frameIndexChanged = pyqtSignal([int])
       
    def __init__(self, parent=None):
        super(TimeLineWidget, self).__init__(parent)

        self.setEnabled(False)        

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.frameIndexChanged)

        self.btn_rewind = QPushButton(QIcon(find('icons/playback-jump-backward.svg')), '')
        self.btn_rewind.clicked.connect(self.rewind)
        self.btn_togglePlay = QPushButton(QIcon(find('icons/playback-play.svg')), '')
        self.btn_togglePlay.clicked.connect(self.togglePlay)
                                  
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(2,2,2,2)
        layout.addWidget(self.slider)
        layout.addWidget(self.btn_rewind)
        layout.addWidget(self.btn_togglePlay)
        self.setLayout(layout)

    def setFrameIndex(self, index):
        self.slider.setValue(index)

    def setFrameCount(self, count):
        self.slider.setMaximum(count)
        self.setEnabled(count)

    def playChanged(self, playing):
        if playing:
            self.btn_togglePlay.setIcon(QIcon(find('icons/playback-pause.svg')))
        else:
            self.btn_togglePlay.setIcon(QIcon(find('icons/playback-play.svg')))
