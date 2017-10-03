Trep Changelog
==============

v1.0.3 (2017-10-03)
-------------------
* Fixed TypeError when visualizing external wrench [#34](https://github.com/MurpheyLab/trep/issues/34)
* Cleaned up a few documentation building errors

v1.0.2 (2015-09-25)
-------------------
* Renamed src/ folder to trep/ following convention [#33](https://github.com/MurpheyLab/trep/issues/33)
* Updated documentation theme to RTD template

v1.0.1 (2015-05-04)
-------------------
* Method added to get trep include path [#31](https://github.com/MurpheyLab/trep/issues/31)

v1.0.0 (2015-04-06)
-------------------
* New public methods exposed in C-API version 3
* Linear damper added [#22](https://github.com/MurpheyLab/trep/issues/22)
* PointToPoint constraint type added
* Named tuples added to DOptimizer returns [#3](https://github.com/MurpheyLab/trep/issues/3)
* Bugfixes and updates to URDF parser

v0.93.1 (2014-11-24)
--------------------
* Fixed a major bug with fixed RZ frame transformations
* Allow URDF tool to be imported without ROS installed
* Added examples for upcoming CISM paper

v0.93.0 (2014-11-13)
--------------------
* Added ROS URDF import tool and documentation
* Fixed a caching bug with structure updates [#19](https://github.com/MurpheyLab/trep/issues/19)

v0.92.1 (2014-07-16)
--------------------
* Fixed a number of small bugs - see github commits for details
* Modified setup.py to be compatible for release through pip
* First release using semantic versioning
