# Demonstrates how to use boto3 to access the lex api.
import json
import boto3
import glob
import pandas as pd

# All configuration variables here.
server_config = './boto3/keys.json'
bot_config = './boto3/bot.json'
input_dir = './input/'
output_dir = './output/'

input_file_paths = input_dir + '*.xlsx'

with open(server_config) as f:
    keys = json.load(f)

# To see a list of available clients, use:
# print(boto3.session.Session().get_available_services())
lex = boto3.client(
    'lex-runtime',
    aws_access_key_id=keys['access_key'],
    aws_secret_access_key=keys['secret_key'],
    region_name=keys['default_region'],
)


def get_msg(bot_config, input_text):
    with open(bot_config) as f:
        bot = json.load(f)
    res = lex.post_text(
        botAlias=bot['botAlias'],
        botName=bot['botName'],
        userId=bot['userId'],
        inputText=input_text
    )
    res_code = res['ResponseMetadata']['HTTPStatusCode']
    if res_code == 200:
        return res['message']
    else:
        raise RuntimeError(f'Server response code: {res_code}')


print(get_msg(bot_config, 'Hello'))







with open(r'./boto3/bot.json') as f:
    bot = json.load(f)

input_text = 'hello'
res = lex.post_text(
    botAlias=bot['botAlias'],
    botName=bot['botName'],
    userId=bot['userId'],
    inputText=input_text
)

if res['ResponseMetadata']['HTTPStatusCode'] == 200:
    print(res['ResponseMetadata'])
