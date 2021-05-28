# aws

## S3

Setting bucket policy: https://docs.aws.amazon.com/AmazonS3/latest/userguide/example-bucket-policies.html#example-bucket-policies-use-case-2

Setting bucket lifecycle config: https://docs.aws.amazon.com/AmazonS3/latest/userguide/how-to-set-lifecycle-configuration-intro.html

Setting replication rules: https://docs.aws.amazon.com/AmazonS3/latest/userguide/replication-example-walkthroughs.html#enable-replication-add-rule

## EFS

You need to ssh to the ec2 instance and enter the mount command under the attach instructions of efs.

Attaching efs to instance instructions: https://docs.aws.amazon.com/efs/latest/ug/mounting-fs-old.html

Of successful, list the instance storage using:

```
sudo df -hT
```

Under the list of filesystems, there should be one similar to the below:

```
fs-19bc09ad.efs.us-east-1.amazonaws.com:/ nfs4      8.0E     0  8.0E   0% /home/
```
