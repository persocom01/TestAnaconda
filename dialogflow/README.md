# dialogflow

A dialogflow python connector testing playground.

## Installation

dialogflow does not require anaconda. Since you probably already have it installed, we are going to use anaconda's ability to create new python environments.

1. Ensure anaconda is installed

* [Anaconda 2019.07](https://www.anaconda.com/distribution/#download-section)

2. Create new anaconda environment

It is recommended that the dialogflow python connector be installed in a new python environment. To create one, open the anaconda prompt and enter:

```
conda create -n dialogflow_env_name python==3.8
```

-n = --name

After which, activate the new environment by entering:

```
conda activate dialogflow_env_name
```

3. Install dependencies

With the new environment activated, install needed dependencies by entering the following:

```
pip install -U pip
pip install dialogflow
```

-y = auto yes
-U = update

# Usage

Get dialogflow lanuage code from the bot. Get project id from the chatbot settings.

Get the dialogflow project key by clicking on the dialogflow project id and going to google cloud console. From there got to IAM_&_Admin>service_accounts>choose_account>add_key>json
