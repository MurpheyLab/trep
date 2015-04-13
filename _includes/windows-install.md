### 1. Install prerequisites

The binaries that are provided for trep on Windows are compiled against recent versions of dependency libraries, therefore current versions of the following dependencies required to be installed prior to installing trep. 64-bit or 32-bit versions of trep are available depending on which Python format is installed.

Download links for [python wheels](https://pip.pypa.io/en/latest/user_guide.html#installing-from-wheels) are provided from [http://www.lfd.uci.edu/~gohlke/pythonlibs/](http://www.lfd.uci.edu/~gohlke/pythonlibs/).  Trep can also be installed on Python distributions such as Anaconda or Canopy provided that the correct dependency versions are installed.

* Python (==2.7.x) - [http://www.python.org/](http://www.python.org/)
  * Download links: [[32-bit]](https://www.python.org/ftp/python/2.7.9/python-2.7.9.msi) [[64-bit]](https://www.python.org/ftp/python/2.7.9/python-2.7.9.amd64.msi)
* Numpy (>=1.8.0) - [http://www.numpy.org/](http://www.numpy.org/) 
  * Download link: [[numpy‑MKL‑1.9.2]](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
* Scipy (>=0.14.0) - [http://www.scipy.org/](http://www.scipy.org/) 
  * Download link: [[scipy-0.15.1]](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)

The following packages are optional. Trep will work fine without them, but they are required to use any of the visualization tools:

* PyOpenGL - [http://pyopengl.sourceforge.net/](http://pyopengl.sourceforge.net/)
  * Download link: [[PyOpenGL-3.1.1]](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl)
* PyQt4 - [http://www.riverbankcomputing.co.uk/software/pyqt/intro](http://www.riverbankcomputing.co.uk/software/pyqt/intro)
  * Download link: [[PyQt4-4.11.3]](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqt4)
* Python Imaging Library - [http://www.pythonware.com/products/pil/](http://www.pythonware.com/products/pil/)
  * Download link: [[Pillow-2.8.1]](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pil)
* matplotlib - [http://matplotlib.sourceforge.net/](http://matplotlib.sourceforge.net/)
  * Download link: [[matplotlib-1.4.3]](http://www.lfd.uci.edu/~gohlke/pythonlibs/#matplotlib)

Please ensure that all dependencies are installed prior to installing Trep.

<br>

### 2. Installing trep

Installers for trep can be downloaded from the [Releases page](https://github.com/MurpheyLab/trep/releases) on Github or with the following links:

* v1.0.0: [[32 bit]](https://github.com/MurpheyLab/trep/releases/download/v1.0.0/trep-1.0.0.win32-py2.7.exe) [[64-bit]](https://github.com/MurpheyLab/trep/releases/download/v1.0.0/trep-1.0.0.win-amd64-py2.7.exe)

Following installation, you can check to see if trep is working by opening a python shell and running

    import trep
    trep.__version__
