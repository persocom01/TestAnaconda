# rasa

A rasa testing playground. Rasa is an open source framework for building AI chatbots. Rasa's official docs can be found here: https://rasa.com/docs/rasa/

## Installation

### Windows

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
pip install --upgrade pip==20.2 --user
pip install rasa
```

-y = auto yes
-U = update
--user = installs pip for the user instead of globally. To install globally, run anaconda as admin instead.

Note that pip is version 20.2. This is because dependency resolution backtracking logic introduced in pip 20.3 makes rasa x install take forever otherwise.

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

It current does not appear to specify a domain for rasa x even though it is listed in the commands here:
https://rasa.com/docs/rasa/command-line-interface/#rasa-x

Use the web browser and navigate to http://localhost:5002/ to access the dashboard.

### Linux

1. Update apt or apt-get

```
RUN apt-get update
RUN apt-get -y upgrade
```

2. Install locale

If writing a Dockerfile:
```
# Set the locale
RUN apt-get -y install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
```

server:
```
sudo locale-gen en_US en_US.UTF-8
sudo dpkg-reconfigure locales
```

You may check the current locale setting at any time by using the command `locale`.

3. Install pip and python3

```
RUN apt-get -y install python3-dev
RUN apt-get -y install python3-pip
pip3 install -U pip
```

4. Install rasa

```
pip install rasa
```

5. Initialize new rasa project

Go to the desired rasa project location, create the project folder and enter it:

```
mkdir project_name
cd project_name
```

Initialize the rasa project.

```
rasa init
```

6. (optional) Install rasa x

Unlike the local version of rasa x, the server version does not need the full bot in the same folder, but only access to the bot's `model` folder.

* Install curl and sudo not already present.

```
apt-get -y install curl
apt-get install sudo
```

* Copy the installation script into the bot directory by entering:

```
curl -sSL -o install.sh https://storage.googleapis.com/rasa-x-releases/0.39.3/install.sh
```

* Run the installation script

The default location for rasa x is `/etc/rasa`. However, this can be undesirable due to permissions. To change the installation path, first set the `RASA_HOME` environmental variable by entering:

```
export RASA_HOME=~/rasa_project_path
```

~ = /home/ubuntu

After which, run the installation script by entering:

```
sudo -E bash ./install.sh
```

-E = use current environment variables

There is a possibility that setting the environmental variable did not work. In such a case, the target folder will have a number of files but be missing others, notably `docker-compose.yml`.

One fix is to open `rasa_x_playbook.yml` and modify the `default` path in the following line:

```
RASA_HOME: "{{ lookup('env','RASA_HOME')|default('rasa_project_path', true) }}"
```

Where rasa_project_path should be the full path of the project folder, without using ~/, for example, `/home/ubuntu/rasa`.

You may need to modify the permissions of the `rasa_x_playbook.yml` file in order to save it. Use `sudo chmod 777 rasa_x_playbook.yml` if necessary.

After which, run the following command, which is the last line in `install.sh`:

```
sudo /usr/local/bin/ansible-playbook -i "localhost," -c local rasa_x_playbook.yml
```

* Run rasa x

Enter:

```
sudo docker-compose up -d
```

At this point rasa x should be accessible at the ip address of the host. If you try to access it too early or it fails to start up, you might see the following json when accessing the ip address:

```
{"database_migration":{"status":"pending","current_revision":[],"target_revision":["97280f5b6803"],"progress_in_percent":0.0}}
```

Where `progress_in_percent` should eventually reach 100% if the images are functioning properly. Once the images have finished starting up, you still need to set the admin password by entering:

```
sudo python3 rasa_x_commands.py create --update admin me <PASSWORD>
```

The rasa x dashboard should now be accessible.

To be continued... https://rasa.com/docs/rasa-x/installation-and-setup/install/docker-compose

## Usage

1. Define domain.

A domain file is needed to define all intents, entities, slots and responses in rasa. By default, the domain file is domain.yml. However, one can use a domain directory instead, where all files will be appended together.

rasa uses .yml files for all training data, but accepts .md for compatibility with legacy versions. README should be written in other formats or they will cause errors during training.

Once the domain is defined, train the model by entering:

```
rasa train -d domain_dir
```

-d = domain. It is not needed if using the default domain.yml file.

2. Define training data.

Training data is defined in the data folder. The files are appended together during training. If domain and responses are already defined, you may use rasa interactive to help write a story for rasa:

```
rasa interactive -d domain_dir
```

3. Test bot.

To test the bot in command line, enter:

```
rasa shell
```

`nlu` can be added to see what the model extracts as intents and entities from text.
`--debug` can be added to help diagnose background processes during training.
