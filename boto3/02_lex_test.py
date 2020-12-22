# Demonstrates how to use boto3 to access the lex api.
# This code is meant to test a lex chatbot. It takes excel files from the input
# folder and outputs them to the output folder.
import json
import boto3
import glob
from pathlib import Path
import pandas as pd

# All configuration variables here.
server_config = './boto3/keys.json'
bot_config = './boto3/bot.json'
input_dir = './boto3/input/'
output_dir = './boto3/output/'
utterances_column = 'Sample Utterances'
result_column = 'Results for ROUND '
number_of_test_runs = 2

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


# Gets the response message for a given input text to the chatbot.
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


# Processes all excel files in the input folder.
files = glob.glob(input_file_paths)

for file in files:
    export_path = output_dir + Path(file).stem + '_result.xlsx'
    data = pd.read_excel(file)
    df = pd.DataFrame(data)
    for i in range(number_of_test_runs):
        col = result_column + str(i + 1)
        df[col] = df[utterances_column].map(lambda x: get_msg(bot_config, x))
    df.to_excel(export_path, index=False)
