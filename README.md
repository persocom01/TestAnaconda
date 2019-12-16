# TestAnaconda

An Anaconda testing playground based on tutorials found here:
* numpy https://www.tutorialspoint.com/numpy/index.htm
* scipy https://www.tutorialspoint.com/scipy/index.htm
* pandas https://www.tutorialspoint.com/python_pandas/index.htm
* matplotlib https://www.tutorialspoint.com/matplotlib/index.htm
* seaborn https://www.tutorialspoint.com/seaborn/index.htm
* sklearn https://www.tutorialspoint.com/machine_learning_with_python/index.htm
* jupyter https://www.tutorialspoint.com/jupyter/index.htm

Workable code is written on each topic to demonstrate use of the python module in the topic area.

## Installation

This project was done on atom within the anaconda environment.

* [https://atom.io/] - atom 1.40.1
* [https://www.anaconda.com/distribution/#download-section] - Anaconda 2019.07

pip was used to install several python modules that helped enhance anaconda.

* pip install graphviz
* pip install modin
* pip install pydotplus
* pip install xgboost

graphviz must also be installed on windows from the following link:

* [https://graphviz.gitlab.io/_pages/Download/Download_windows.html] - graphviz 2.38

The following keys must then be added to the windows environmental PATH variable:

* C:\\Program Files (x86)\\Graphviz2.38\\bin
* C:\\Program Files (x86)\\Graphviz2.38\\bin\\dot.exe

After installation of atom and anaconda, open atom within the anaconda environment by opening the anaconda prompt and typing:

atom

### Atom packages used:

* Hydrogen
* atom-ide-debugger-python
* linter-flake8
* python-autopep8
* python-debugger

### General packages:

* atom-beautify
* busy-signal
* file-icons
* intentions
* minimap
* open_in_cmd
* project-manager
* script
