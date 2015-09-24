from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np


class VisualScene(QObject):

    frameCountChanged = pyqtSignal([int])
    frameTimeChanged = pyqtSignal([float])
    frameIndexChanged = pyqtSignal([int])
    itemsChanged = pyqtSignal()
    playChanged = pyqtSignal([bool])
    
    def __init__(self):
        super(VisualScene, self).__init__()

        self._items = []

        self._frameTimes = np.array([])
        self._frameIndex = 0

        self._playing = False
        self._playback_period = 40 # Frame period in ms
        self._playbackTimer = None


    def frameCount(self):
        return len(self._frameTimes)


    def time(self):
        if self.frameCount():
            return self._frameTimes[self._frameIndex]
        else:
            return 0.0


    def frameIndex(self):
        return self._frameIndex


    def setFrameIndex(self, index):
        if self.frameCount() == 0:
            index = 0
        index = min(index, self.frameCount() - 1)
        index = max(index, 0)
        if index != self._frameIndex:
            self._frameIndex = index
            self.frameIndexChanged.emit(self._frameIndex)
            self.frameTimeChanged.emit(self.time())

            
    @property
    def frameRate(self):
        return 1000.0 / self._playback_period


    @frameRate.setter
    def frameRate(self, value):
        self._playback_period = int(1000.0/value)
        self.updateTimeRange()


    def addItem(self, item):
        self._items.append(item)
        self.updateTimeRange()
        self.itemsChanged.emit()

    def items(self):
        return self._items


    def updateTimeRange(self):
        """
        Calculates the minimum and maximum times from the current
        content.  Emits timeRangeChanged()
        """
        t0 = None
        tf = None

        for item in self._items:
            info = item.getTimeRange()
            if info is None:
                continue
            if t0 is None or t0 > info[0]:
                t0 = info[0]
            if tf is None or tf > info[1]:
                tf = info[1]

        if t0 is None:
            self._frameTimes = np.array([])
            self.setFrameIndex(0)
            self.frameCountChanged.emit(0)
        else:
            self._frameTimes = np.arange(t0, tf, self._playback_period/1000.0)
            self.setFrameIndex(0)
            self.frameCountChanged.emit(self.frameCount())


    def play(self):
        if self._playing:
            return
        self._playing = True
        if self._playbackTimer is None:
            self._playbackTimer = self.startTimer(self._playback_period)
        self.playChanged.emit(self._playing)

    
    def pause(self):
        if not self._playing:
            return
        self._playing = False
        if self._playbackTimer is not None:
            self.killTimer(self._playbackTimer)
            self._playbackTimer = None
        self.playChanged.emit(self._playing)


    def togglePlay(self):
        if self._playing:
            self.pause()
        else:
            self.play()


    def rewind(self):
        self.pause()
        self.setFrameIndex(0)
    
  
    def playbackTimerEvent(self, event):
        if self.frameIndex() == self.frameCount() - 1:
            self.pause()
        else:
            self.setFrameIndex(self.frameIndex() + 1)


    def timerEvent(self, event):
        if event.timerId() == self._playbackTimer:
            return self.playbackTimerEvent(event)
