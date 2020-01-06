# TestAnaconda

An Anaconda testing playground. Tutorials for various modules found here:
* [https://www.tutorialspoint.com/numpy/index.htm](numpy)
* [https://www.tutorialspoint.com/scipy/index.htm](scipy)
* [https://www.tutorialspoint.com/python_pandas/index.htm](pandas)
* [https://www.tutorialspoint.com/matplotlib/index.htm](matplotlib)
* [https://www.tutorialspoint.com/seaborn/index.htm](seaborn)
* [https://www.tutorialspoint.com/machine_learning_with_python/index.htm](sklearn)
* [https://flask.palletsprojects.com/en/1.1.x/](flask)
* [https://www.tutorialspoint.com/jupyter/index.htm](jupyter)

Workable code is written on each topic to demonstrate use of the python module in the topic area.

## Installation

This project was done on atom within the anaconda environment.

* [https://atom.io/](atom_1.40.1)
* [https://www.anaconda.com/distribution/#download-section](Anaconda_2019.07)

pip/conda install was used to add several python modules to anaconda. Open the anaconda prompt in admin mode and type:

```
pip install graphviz
conda install -c conda-forge imbalanced-learn
pip install kaggle
pip install modin
pip install pmdarima
pip install pydotplus
pip install xgboost
conda install pysal
pip install folium
```

They following packages will be installed:

* graphviz - for drawing graphs in DOT language scripts.
* imbalanced-learn - for dealing with unbalanced classes.
* kaggle - for use of the kaggle API for kaggle competition submissions.
* modin - for multi-core pandas.
* pmdarima - adds AutoARIMA to python's time series analysis capability.
* pydotplus - python's interface to graphviz's DOT language.
* xgboost - eXtreme Gradient Boosting.
* pysal - for analysis of geospatial data.
* folium - for interactive map visualizations.

graphviz must also be installed on windows from the following link:

* [https://graphviz.gitlab.io/_pages/Download/Download_windows.html](graphviz_2.38)

The following keys must then be added to the windows environmental PATH variable:

```
C:\Program Files (x86)\Graphviz2.38\bin
C:\Program Files (x86)\Graphviz2.38\bin\dot.exe
```

After installation of atom and anaconda, open atom within the anaconda environment by opening the anaconda prompt and typing:

```
atom
```

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
