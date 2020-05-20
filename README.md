# TestAnaconda

An Anaconda testing playground. Tutorials for various modules found here:

| Package | Tutorial |
| ------ | ------ |
| numpy | https://www.tutorialspoint.com/numpy/index.htm |
| scipy | https://www.tutorialspoint.com/scipy/index.htm |
| pandas | https://www.tutorialspoint.com/python_pandas/index.htm |
| matplotlib | https://www.tutorialspoint.com/matplotlib/index.htm |
| seaborn | https://www.tutorialspoint.com/seaborn/index.htm |
| sklearn | https://www.tutorialspoint.com/machine_learning_with_python/index.htm |
| flask | https://flask.palletsprojects.com/en/1.1.x/ |
| jupyter | https://www.tutorialspoint.com/jupyter/index.htm |

Workable code is written on each topic to demonstrate use of the python module in the topic area.

## Installation

Anaconda has to be downloaded and installed. Atom was used as text editor.

* [Anaconda 2019.07](https://www.anaconda.com/distribution/#download-section)
* [atom 1.40.1](https://atom.io/)

pip/conda install was used to add modules to anaconda. Open the anaconda prompt in admin mode and type:

```
pip install boto3
pip install folium
pip install graphviz
conda install -c conda-forge imbalanced-learn
pip install kaggle
pip install modin
pip install pmdarima
pip install ppscore
pip install pydotplus
conda install pysal
conda install -c conda-forge selenium
conda install -c conda-forge wordcloud
pip install xgboost
```

This installs following modules:

* boto3 - python connector for Amazon Web Services.
* folium - for interactive map visualizations.
* graphviz - for drawing graphs in DOT language scripts.
* imbalanced-learn - for dealing with unbalanced classes.
* kaggle - for use of the kaggle API for kaggle competition submissions.
* modin - for multi-core pandas.
* pmdarima - adds AutoARIMA to python's time series analysis capability.
* ppscore - an alternative to pandas' df.corr() that works across continuous and categorical variables and is asymmetric. It does, however, take potentially much longer to compute.
* pydotplus - python's interface to graphviz's DOT language.
* pysal - for analysis of geospatial data.
* selenium - for scraping web pages.
* wordcloud - for generation of wordclouds for NLP visualization.
* xgboost - eXtreme Gradient Boosting.

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

To use anaconda nlp modules, there is also a need to run the following as a python file:

```
import nltk
nltk.download()
```

A window will pop up, where desired nlp packages can be selected.

### Updating

Open the anaconda prompt in admin mode and type:
```
conda update -n root conda
```

After which, update individual environments using:
```
conda update --all
```

### Atom packages used:

* Hydrogen
* linter-flake8
* python-autopep8

### General packages:

* atom-beautify
* busy-signal
* file-icons
* intentions
* minimap
* open_in_cmd
* project-manager
* script
