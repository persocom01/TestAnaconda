# Flask

A flask app testing playground.

## Installation

### Local machine

It seems to be recommended that you install flask and all the packages the app needs in a separate environment, which is what venv is for. Flask can be run without installing the virtual environment.
Open the project folder in cmd and type:

```
python -m venv env_name
cd env_name/Scripts/activate
pip install flask
```

After which, create a python file with the following minimalist code:

```
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(debug=True)
```

Do not set debug=True if using it as a production environment.

### AWS

1. Spin up an aws ec2 instance. Normally linux or ubuntu AMI.

2. Download the .pem key.

3. git bash (install on computer if not already present) in the folder with the key and type:

```
chmod 400 keyname.pem
```

which gives the user permission to read the file (4) and no permissions (0) to the group and everyone else.

4. Connect to the aws instance using the following command:

```
ssh -i keyname.pem username@aws_instance_public_dns
```

The list of user names is as follows:

| OS | Official AMI ssh Username |
| ------ | ------ |
| Amazon Linux | ec2-user |
| Ubuntu | ubuntu |
| Debian | admin |
| RHEL 6.4 and later | ec2-user |
| RHEL 6.3 and earlier | root |
| Fedora | fedora |
| Centos | centos |
| SUSE | ec2-user |
| BitNami | bitnami |
| TurnKey | root |
| NanoStack | ubuntu |
| FreeBSD | ec2-user |
| OmniOS | root |

To remove the added ip from the known hosts list, use:

```
ssh-keygen -R server_ip_address
```

To end the connection, enter:

```
exit
```

## Usage

### Local machine

1. Open the project folder in cmd and type:

```
env_name\Scripts\activate
set FLASK_APP=test_app
set FLASK_ENV=development
any_custom_flask_commands
flask run
```

2. Use the shell script by opening the project folder in cmd and typing:

start_flask.sh

### AWS

In order to allow a shell script to be run on linux, the file must be given the appropriate permissions. Do so with the following code:

```
sudo chmod 755 start_flask.sh
```
