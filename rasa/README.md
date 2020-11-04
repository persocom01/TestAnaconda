# rasa

A rasa testing playground. Rasa is an open source framework for building AI chatbots. Rasa's official docs can be found here: https://rasa.com/docs/rasa/

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

5. (optional) Install rasa x

rasa x is a dashboard like browser interface with which to interact with and share rasa. Install it by entering:

```
pip install rasa-x --extra-index-url https://pypi.rasa.com/simple
```

Run it by navigating to the rasa bot folder and entering:

```
rasa x
```

Use the web browser and navigate to http://localhost:5002/ to access the dashboard.

## Usage

1. Define domain.

A domain file is needed to define all intents, entities, slots and responses in rasa. By default, the domain file is domain.yml. However, one can use a domain directory instead, where all files will be appended together. Once defined, train the model on the new domain by entering:

```
rasa train -d domain_dir
```

-d = domain. It is not needed if using the default domain.yml file.

2. Define training data.

Training data is defined in the data folder. The files are appended together during training.

3. Test bot.

To test the bot in command line, enter:

```
rasa shell
```
