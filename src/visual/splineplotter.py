from PyQt4.QtCore import *
from PyQt4.QtGui import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure


class SplinePlotter(QDialog):
    def __init__(self, spline, points=1000, parent=None):
        super(SplinePlotter, self).__init__(parent=parent)

        self.fig = Figure()
        self.y_axes = self.fig.add_subplot(3,1,1)
        self.dy_axes = self.fig.add_subplot(3,1,2)
        self.ddy_axes = self.fig.add_subplot(3,1,3)

        self.plot_widget = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.plot_widget, self)

        panel = QVBoxLayout()
        panel.addWidget(self.toolbar)
        panel.addWidget(self.plot_widget)        
        self.setLayout(panel)


        xmin = min(spline.x_points)
        xmax = max(spline.x_points)

        spline = spline
        x = [ xmin + float(i)/points*(xmax-xmin) for i in range(points) ]
        y = [spline.y(xi) for xi in x]
        dy = [spline.dy(xi) for xi in x]
        ddy = [spline.ddy(xi) for xi in x]

        self.y_axes.plot(spline.x_points, spline.y_points, 'o')
        self.y_axes.plot(x, y)
        self.dy_axes.plot(x,dy)
        self.ddy_axes.plot(x, ddy)


def plot_spline(spline, points=1000):
    app = QApplication([])
    plotter = SplinePlotter(spline, points)
    plotter.show()
    app.exec_()

    
    
