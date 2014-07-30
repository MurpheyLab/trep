### 1. Install prerequisites

Trep requires that the following basic dependencies are installed:

* Python - [http://www.python.org/](http://www.python.org/) (>=2.6) (including development header files)
* Numpy - [http://www.numpy.org/](http://www.numpy.org/) (>=1.4.1)
* Scipy - [http://www.scipy.org/](http://www.scipy.org/)

You also need a C compiler installed and configured properly to compile Python extensions.

To install the basic prerequisites, run the following command:

    sudo aptitude install python python-dev python-numpy python-scipy


The following packages are optional. Trep will work fine without them, but they are required to use any of the visualization tools:

* PyOpenGL - [http://pyopengl.sourceforge.net/](http://pyopengl.sourceforge.net/)
* PyQt4 - [http://www.riverbankcomputing.co.uk/software/pyqt/intro](http://www.riverbankcomputing.co.uk/software/pyqt/intro)
* Python Imaging Library - [http://www.pythonware.com/products/pil/](http://www.pythonware.com/products/pil/)
* matplotlib - [http://matplotlib.sourceforge.net/](http://matplotlib.sourceforge.net/)

To install the all prerequisites including visualizations, run the following command:

    sudo aptitude install python python-dev python-opengl python-numpy python-scipy python-imaging \
     python-qt4 python-qt4-gl python-matplotlib freeglut3-dev

<br>

### 2.1. Installing with pip

Trep can be installed from the Python Package Index using [pip](https://pip.pypa.io/en/latest/index.html).  If you have already [installed pip](https://pip.pypa.io/en/latest/installing.html), install trep by running the following command

    sudo pip install trep

<br>

### 2.2. Installing from source

Checkout the development version of trep from Github using the following

    git clone -b master https://github.com/MurpheyLab/trep.git

Build trep with the following commands

    cd trep
    python setup.py build

After the compilation finishes, install trep with

    sudo python setup.py install
