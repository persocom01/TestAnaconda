# Boto3

Boto3 is used to connect to Amazon Web Services.

## Installation

To install boto3, enter the following into cmd:

```
pip install boto3
```

## Setup

Before Boto3 can be used, one must first setup Amazon authentication credentials.

This can be done by logging in> My security credentials> AWS IAM credentials> Create access key

The default region can be found here: https://docs.aws.amazon.com/general/latest/gr/rande.html

After which, setup the client inside code like this:

```
s3 = boto3.client(
    's3',
    aws_access_key_id=keys['access_key'],
    aws_secret_access_key=keys['secret_key'],
    region_name=keys['default_region'],
)
```
