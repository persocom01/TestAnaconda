# rasa

A rasa testing playground.

## Installation

1. Ensure anaconda and Microsoft VC++ Compiler are installed

* [Anaconda 2019.07](https://www.anaconda.com/distribution/#download-section)
* [Microsoft VC++ Compiler](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)

2. Create new anaconda environment

It is recommended that rasa be installed in a new anaconda environment. To create one, open the anaconda prompt and enter:

```
conda create -n rasa_env_name python==3.7.9
```

-n = --name
While rasa currently works with python 3.8, some of the dependencies still use 3.7, so we'll go with that.

After which, activate the new environment by entering:

```
conda activate rasa_env_name
```

3. Install dependencies

With the new environment activated, install needed dependencies by entering the following:

```
conda install -y ujson
conda install -y tensorflow
pip install -U pip
pip install rasa
```

-y = auto yes
-U = update

4. Initialize new rasa project

Navigate to the desired project folder in explorer, right click the address bar, and paste it into the anaconda prompt like this:

```
cd paste_address
```

Initialize the rasa project.

```
rasa init
```

## Usage

1. Define domain.

A domain file is needed to define all intents, entities, slots and responses in rasa. By default, the domain file is domain.yml. However, one can use a domain directory instead, where all files will be appended together. Once defined, train the model on the new domain by entering:

```
rasa train -d domain_dir
```

-d = domain. It is not needed if using the default domain.yml file.

2. Define training data.

Training data is defined in the data folder. The files are appended together so there is

2. Test bot.

To test the bot in command line, enter:

```
rasa shell
```




pip/conda install was used to add modules to anaconda. Open the anaconda prompt in admin mode and type:

```
pip install boto3
pip install folium
pip install graphviz
conda install -c conda-forge imbalanced-learn
pip install kaggle
conda install -c conda-forge lightgbm
pip install modin
conda install -c conda-forge opencv
pip install pmdarima
pip install ppscore
pip install pydotplus
conda install pysal
conda install -c conda-forge selenium
conda install -c conda-forge wordcloud
conda install -c anaconda py-xgboost
```

This installs following modules:

* boto3 - python connector for Amazon Web Services.
* folium - for interactive map visualizations.
* graphviz - for drawing graphs in DOT language scripts.
* imbalanced-learn - for dealing with unbalanced classes.
* kaggle - for use of the kaggle API for kaggle competition submissions.
* lightgbm - LightGBM gradient boosting framework.
* modin - for multi-core pandas.
* opencv - Open Source Computer Vision library.
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

## Known issues

1. python's locale.getpreferredencoding() returns cp1252 in windows. This may cause problems with information from web apis. To rectify this problem insert the following code on top of python files with encoding issues:

```
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
```

2. pandas appears to use utf-8 encoding by default. This may cause problems when the encoding is actually cp1252. To rectify this problem set encoding='cp1252' when reading the file.
