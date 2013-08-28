from PyQt4.QtCore import *
from PyQt4.QtGui import *


class View2D(QWidget):

    def __init__(self, parent=None):
        super(View2D, self).__init__(parent)
        self._scene = None

        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._scale_exponent = 0

        self._lastPos = None
        self._center = QPoint(0, 0)


    def setScene(self, scene):
        self._scene = scene
        self._scene.frameTimeChanged.connect(self.update)
        self._scene.itemsChanged.connect(self.update)


    def scene(self):
        return self._scene


    def set_scale(self, scale):
        self._scale_exponent = scale
        self.update()


    def set_center(self, center):
        self._center = QPoint(center[0], center[1])
        self.update()


    def wheelEvent(self, event):
        degrees = event.delta()/8.0
        self._scale_exponent += degrees/60.0
        self.update()      


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._lastPos = event.pos()
        else:
            super(View2D, self).mousePressEvent(event)


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._lastPos = None
        else:
            super(View2D, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self._lastPos is not None:
            delta  = event.pos() - self._lastPos
            self._lastPos = event.pos()

            self._center = self._center + delta
            self.update()


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), QBrush(Qt.white))

        result = QTransform()

        rect = painter.window()
        result.translate(rect.width()/2.0, rect.height()/2.0)
        result.translate(self._center.x(), self._center.y())
        scale = 0.5**(self._scale_exponent)
        result.scale(scale*rect.height()/2.0, -scale*rect.height()/2.0)

        painter.setTransform(result)

        time = self.scene().time()

        for item in self.scene().items():
            item.setTime(time)
            item.draw(painter)


    def screenshot(self):
        return QPixmap.grabWidget(self)
