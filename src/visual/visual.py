import trep
import numpy as np
import os.path
from scipy.interpolate import interp1d
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from timelinewidget import TimeLineWidget
from visualscene import VisualScene
from view2d import View2D
from view3d import View3D


class BasicViewer(QMainWindow):

    def __init__(self, parent=None):
        super(BasicViewer, self).__init__(parent)

        self.scene = VisualScene()
        
        self.createTimeLineWidget()
        self.view = None


        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        #layout.addWidget(self.view)
        layout.addWidget(self.time_control)
        self.layout = layout

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.createActions()
        self.createMenus()
        #self.createToolBars()
        #self.createStatusBar()

    def createTimeLineWidget(self):
        self.time_control = TimeLineWidget()
        self.time_control.togglePlay.connect(self.scene.togglePlay)
        self.time_control.rewind.connect(self.scene.rewind)
        self.time_control.frameIndexChanged.connect(self.scene.setFrameIndex)
        self.scene.frameIndexChanged.connect(self.time_control.setFrameIndex)
        self.scene.frameCountChanged.connect(self.time_control.setFrameCount)

    def setSceneView(self, view):
        self.view = view
        self.layout.insertWidget(0, view)
        self.view.setScene(self.scene)
        self.view.setFocus(Qt.OtherFocusReason)
   
    def createActions(self):

        self.screenshotAct = QAction("&Save Screenshot...", self,
                                     shortcut="Ctrl+S",
                                     statusTip="Save a screenshot",
                                     triggered=self.screenshot)

        self.allScreenshotsAct = QAction("Save &All Screenshots...", self,
                                     statusTip="Save screenshots for every animation frame.",
                                     triggered=self.screenshots)

        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                               statusTip="Exit the application", triggered=self.close)

        self.rewindAct = QAction("&Rewind", self, shortcut="Ctrl+R",
                                 statusTip="Rewind", triggered=self.rewind)
        
        self.playAct = QAction("&Play/Pause", self, shortcut="Ctrl+P",
                                 statusTip="Play or Pause", triggered=self.playpause)

        self.getCameraAct = QAction("Camera Position", self,
                                    statusTip="Display the Camera position/location",
                                    triggered=self.show_camera_pos)
        
        self.aboutAct = QAction("&About trep", self,
                statusTip="Information about trep",
                triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
                statusTip="Show the Qt library's About box",
                triggered=qApp.aboutQt)


    def createMenus(self):
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenu.addAction(self.screenshotAct)
        self.fileMenu.addAction(self.allScreenshotsAct)
        self.fileMenu.addSeparator();
        self.fileMenu.addAction(self.exitAct)

        self.playMenu = self.menuBar().addMenu("&Playback")
        self.playMenu.addAction(self.rewindAct)
        self.playMenu.addAction(self.playAct)

        self.viewMenu = self.menuBar().addMenu("&View")
        self.viewMenu.addAction(self.getCameraAct)

        self.helpMenu = self.menuBar().addMenu("&Help")
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)


    def rewind(self):
        self.scene.rewind()

    def playpause(self):
        self.scene.togglePlay()

    def show_camera_pos(self):
        try:
            info = ""
            info = "3D camera info:<br>"
            info += "position=%r<br>" % (self.view.camera._pos,)
            info += "angles=%r" % (self.view.camera._ang, )
            QMessageBox.about(self, "Camera Info", info)
            return
        except AttributeError:
            pass
        try:
            info = ""
            info = "2D camera info:<br>"
            info += "xpos=%r<br>" % (self.view._center.x(),)
            info += "ypos=%r<br>" % (self.view._center.y(),)
            info += "scale=%r" % (self.view._scale_exponent, )
            QMessageBox.about(self, "Camera Info", info)
            return
        except AttributeError:
            pass

    def about(self):
        QMessageBox.about(self, "About trep",
                "trep %s<br><Br>" 
                "http://trep.googlecode.com/" % trep.__version__)

    def screenshot(self):
        
        
        filename = QFileDialog.getSaveFileName(self)
        if not filename:
            return
        screenshot = self.view.screenshot()
        screenshot.save(filename)

    def screenshots(self):

        filename = QFileDialog.getSaveFileName(self)
        if not filename:
            return


        root, ext = os.path.splitext(str(filename))

        self.scene.rewind()
        index = self.scene.frameIndex()
        for i in range(self.scene.frameCount()):
            self.scene.setFrameIndex(i)       
            screenshot = self.view.screenshot()
            #screenshot.save(filename)
            screenshot.save('%s-%04d%s' % (root, i, ext))
        self.scene.setFrameIndex(index)

    

def visualize_2d(items, argv=[], center=None, scale=None):
    app = QApplication(argv)
    viewer = BasicViewer()

    viewer.setSceneView(View2D())

    if center is not None:
        viewer.view.set_center(center)
    if scale is not None:
        viewer.view.set_scale(scale)
      
    for item in items:
        viewer.scene.addItem(item)
    
    viewer.show()
    viewer.resize(800, 800)

    return app.exec_()


def visualize_3d(items, argv=[], camera_pos=None, camera_ang=None):
    app = QApplication(argv)
    viewer = BasicViewer()

    viewer.setSceneView(View3D())

    if camera_ang is not None:
        viewer.view.camera._ang = np.array(camera_ang)
    if camera_pos is not None:
        viewer.view.camera._pos = np.array(camera_pos)
      
    for item in items:
        viewer.scene.addItem(item)
    
    viewer.show()
    viewer.resize(800, 800)

    return app.exec_()




