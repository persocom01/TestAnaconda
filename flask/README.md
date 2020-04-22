# Flask tutorial

A flask app testing playground.

## Installation

### Local machine

Open the project folder in cmd and type:

python -m venv env_name

env_name\\Scripts\\activate

pip install flask

### Amazon

1. Spin up an aws instance. Normally linux or ubuntu AMI.

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

## Running

### Local machine

1. Open the project folder in cmd and type:

env_name\\Scripts\\activate

set FLASK_APP=test_app

set FLASK_ENV=development

any_custom_flask_commands

flask run

2. Use the shell script by opening the project folder in cmd and typing:

start_flask.sh
