# S3 ----------------------------------------------------------------------------------------------

# Setup clinet
s3 = boto3.client('s3', region_name='us-east-1', 
                        aws_access_key_id=AWS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET)
# List buckets
bucket_response = s3.list_buckets()
buckets = bucket_response["Buckets"]

# Create and delete buckets
bucket = s3.create_bucket(Bucket="bucket_name")
response = s3.delete_bucket(Bucket="bucket_name")

# Uploading and downloading files
s3.upload_file(Filename="file.csv", Bucket="mybucket", Key="file_name_on_s3.csv")
s3.download_file(Filename="file.csv", Bucket="mybucket", Key="file_name_on_s3.csv")

# List objects in a bucket
response = s3.list_objects(Bucket="mybucket", MaxKeys=2, Preffix="only_files_starting_with_this_string")

# Get object's metadata (modified time, size in bytes etc.)
response = s3.head_object(Bucket="mybucket", Key="file.csv")

# Delete object
s3.delete_object(Bucket="mybucket", Key="file_name_on_s3.csv")

# Make object publicly available or private using ACLs (access control lists)
# Existing object:
s3.put_object_acl(Bucket="mybucket", Key="file.csv", ACL="public-read")
# On upload:
s3.upload_file(Filename="file.csv", Bucket="mybucket", Key="file_name_on_s3.csv", 
               ExtraArgs={"ACL": "private"})

# Downloading objects
response = s3.get_object(Bucket="mybucket", Key="file.csv")
pd.read_csv(response["Body"])

# Accessing private files with pre-signed URLs (expire after some time)
share_url = s3.generate_presigned_url(ClientMethod="get_object", ExpiresIn=3600,
                                      Params={"Bucket": "mybucket", "Key": "file.csv"})
pd.read_csv(share_url)

# Load multiple data files info one data frame
df_list = []
response = s3.list_objects(Bucket="mybucket", Preffix="someprefix")
request_files = response["Contents"]
for file in request_files:
    obj = s3.get_object(Bucket="mybucket", Key=file["Key"])
    obj_df = pd.read_csv(obj["Body"])
    df_list.append(obj_df)
df = pd.concat(df_list)


# SNS ---------------------------------------------------------------------------------------------

# Simple Notification Service, allows to: send emails, SMS messages, push notofications, 
# Publishers push messages to topics, subscribers receive them

# Setup clinet
sns = boto3.client("sns", region_name="us-east-1", 
                          aws_access_key_id=AWS_KEY_ID, 
                          aws_secret_access_key=AWS_SECRET)

# Create topic
response = sns.create_topic(Name="topic_name")
# Amazon resource number, a unique id
topic_arn = response["TopicArn"]

# List topics
response = sns.list_topics()
response["Topics"]

# Deleting topics
sns.delete_topic(TopicArn="arn")

# Create SMS subscription
response = sns.subscribe(TopicArn="arn", Protocol="SMS", Endpoint="+48123456789")
subs_arn = response["SubscriptionArn"]

# Create email subscription
response = sns.subscribe(TopicArn="arn", Protocol="email", Endpoint="user@server.com")
subs_arn = response["SubscriptionArn"]

# Listing subscriptions
response = sns.list_subscriptions()
response = sns.list_subscriptions_by_topic(TopicArn="arn")
subs = response["Subscriptions"]

# Deleting subscription
sns.unsubscribe(SubscriptionArn=subs_arn)

# Deleting multiple subscriptions
for sub in subs:
	if sub["Protocol"] == "sms":
		sns.unsubscribe(sub["SubscriptionArn"])

# Publishing to topics
sns.publish(TopicArn="arn", Message="mess", Subject="sub for email")

# Sending single SMS (no topic, no subscription)
response = sns.publish(PhoneNumber="123", Message="mess")


# Rekognition -------------------------------------------------------------------------------------

# Setup clinet
rekog = boto3.client("rekognition", region_name="us-east-1", 
                        		    aws_access_key_id=AWS_KEY_ID, 
                          		    aws_secret_access_key=AWS_SECRET)

# Object detection
response = rekog.detect_labels(
    Image={'S3Object': {'Bucket': 'mybucket', 'Name': 'picname.jpg'}}, 
    MaxLabels=1
)

# Text detection
response = rekog.detect_text(Image={'S3Object': {'Bucket': 'mybucket', 'Name': 'picname.jpg'}})


# Translate ---------------------------------------------------------------------------------------

# Setup clinet
translate = boto3.client("translate", region_name="us-east-1", 
                        		      aws_access_key_id=AWS_KEY_ID, 
                          		      aws_secret_access_key=AWS_SECRET)

# Translate text to spanish
response = translate.translate_text(
	Text="some text",
	SourceLanguageCode="auto",
	TargetLanguageCode="es"
)


# Comprehend --------------------------------------------------------------------------------------

# Setup clinet
comprehend = boto3.client("comprehend", region_name="us-east-1", 
                        		        aws_access_key_id=AWS_KEY_ID, 
                          		        aws_secret_access_key=AWS_SECRET)

# Detect language
comprehend.detect_dominant_language(Text="some text")

# Detect sentiment
comprehend.detect_sentiment(Text="some text", LanguageCode="en")

