import json
import boto3
import pleiades as ple

with open(r'.\boto3\keys.json') as f:
    keys = json.load(f)

s3 = boto3.resource(
    's3',
    aws_access_key_id=keys['access_key'],
    aws_secret_access_key=keys['secret_key'],
    region_name=keys['default_region'],
)


cz = ple.CZ(s3)

bucket = 'inp-dil-ap-southeast-1'
keys = cz.get_keys(bucket, prefix='/data/eICU/')
savein = './boto3/eicu/'
print(cz.download_files(bucket, keys, savein=savein))
