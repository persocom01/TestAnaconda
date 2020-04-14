import json
import boto3

with open(r'.\boto3\keys.json') as f:
    keys = json.load(f)

s3 = boto3.resource(
    's3',
    aws_access_key_id=keys['access_key'],
    aws_secret_access_key=keys['secret_key'],
    region_name=keys['default_region'],
)


def keys(bucket_name, prefix='/', delimiter='/'):
    prefix = prefix[1:] if prefix.startswith(delimiter) else prefix
    bucket = boto3.resource('s3').Bucket(bucket_name)
    return (_.key for _ in bucket.objects.filter(Prefix=prefix))

print(keys('inp-dil-ap-southeast-1'))

# file_path = './boto3/files/d_cpt.csv.gz'
# s3.download_file('inp-dil-ap-southeast-1', file_path, '/data/MIMIC/D_CPT.csv.gz')

# print(s3.list_buckets())
