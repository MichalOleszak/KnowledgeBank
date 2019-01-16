# Importing flat files from the web ------------------------------------------------------------------------------------
from urllib.request import urlretrieve
import pandas as pd
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
urlretrieve(url, 'winequality-red.csv')
df = pd.read_csv('winequality-red.csv', sep=';')


# Loading files from the web without saving them locally ---------------------------------------------------------------
import matplotlib.pyplot as plt
df = pd.read_csv(url, sep=';')
# Plot first column of df
pd.DataFrame.hist(df.ix[:, 0:1])
plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
plt.ylabel('count')
plt.show()


# Importing non-flat files from the web --------------------------------------------------------------------------------
url = 'http://s3.amazonaws.com/assets.datacamp.com/course/importing_data_into_r/latitude.xls'
xl = pd.read_excel(url, sheetname=None)     # Read in all sheets of Excel file: xl
print(xl.keys())                            # Print the sheetnames
print(xl['1700'].head())                    # Print the head of the first sheet (using its name, NOT its index)


# HTTP requests to import files from the web ---------------------------------------------------------------------------
# Using urllib package
from urllib.request import urlopen, Request
url = "http://www.datacamp.com/teach/documentation"
request = Request(url)         # This packages the request
response = urlopen(request)    # Sends the request and catches the response: response
print(type(response))          # Print the datatype of response
html = response.read()         # Extract the response
response.close()               # Be polite and close the response!

# Using the higher-level requests library
import requests
url = "http://www.datacamp.com/teach/documentation"
r = requests.get(url)          # Packages the request, send the request and catch the response in one go
text = r.text                  # Extract the response

# Parsing HTML with BeautifulSoup
from bs4 import BeautifulSoup
url = 'https://www.python.org/~guido/'
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc)
pretty_soup = soup.prettify()

guido_title = soup.title
guido_text = soup.get_text()

# Extract URLs
# Find all 'a' tags (which define hyperlinks): a_tags
a_tags = soup.find_all("a")
# Print the URLs to the shell
for link in a_tags:
    print(link.get('href'))


# Interacting with APIs to import data from the web --------------------------------------------------------------------
# Loading and exploring a JSON
with open("a_movie.json") as json_file:
    json_data = json.load(json_file)
# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])

# API requests - Open Movie Database
import requests
url = 'http://www.omdbapi.com/?apikey=ff21610b&t=social+network'
r = requests.get(url)
print(r.text)
json_data = r.json()
for k in json_data.keys():
    print(k + ': ', json_data[k])

# Wikipedia API
url = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza'
r = requests.get(url)
json_data = r.json()
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)

# Twitter API
import tweepy
# Store OAuth authentication credentials in relevant variables (Twitter account needed)
access_token = "1092294848-aHN7DcRP9B4VMTQIhwqOYiB14YkW92fFO8k8EPy"
access_token_secret = "X4dHmhPfaksHcQ7SCbmZa2oYBBVSD2g8uIHXsp5CTaksx"
consumer_key = "nZ6EA0FxZ293SxGNg8g8aP0HM"
consumer_secret = "fJGEodwe3KiKUnsYJC3VRndj7jevVvXbK2D5EiJ2nehafRgA6i"
# Pass OAuth details to tweepy's OAuth handler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Define a Tweet listener that creates a file called 'tweets.txt', collects streaming tweets as .jsons  and writes them
# to the file 'tweets.txt'; once 100 tweets have been streamed, the listener closes the file and stops listening.
class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open("tweets.txt", "w")

    def on_status(self, status):
        tweet = status._json
        self.file.write( json.dumps(tweet) + '\n' )
        self.num_tweets += 1
        if self.num_tweets < 100:
            return True
        else:
            return False
        self.file.close()

    def on_error(self, status):
print(status)


# Initialize Stream listener
l = MyStreamListener()
# Create you Stream object with authentication
stream = tweepy.Stream(auth, l)
# Filter Twitter Streams to capture data by the keywords:
stream.filter(track=['clinton', 'trump', 'sanders', 'cruz'])

tweets_data_path = 'tweets.txt'


# Initialize empty list to store tweets:
tweets_data = list()
# Open connection to file
tweets_file = open(tweets_data_path, "r")
# Read in tweets and store in list:
for line in tweets_file:
    tweet = json.loads(line)
    tweets_data.append(tweet)
# Close connection to file
tweets_file.close()
# Print the keys of the first tweet dict
print(tweets_data[0].keys())

df = pd.DataFrame(tweets_data, columns=['text', 'lang'])

# Initialize list to store tweet counts
[clinton, trump, sanders, cruz] = [0, 0, 0, 0]

# Function to check if  the first argument (a word) occurs within the 2nd argument (a tweet)
import re

def word_in_text(word, tweet):
    word = word.lower()
    text = tweet.lower()
    match = re.search(word, tweet)

    if match:
        return True
    return False

# Iterate through df, counting the number of tweets in which each candidate is mentioned
for index, row in df.iterrows():
    clinton += word_in_text('clinton', row['text'])
    trump += word_in_text('trump', row['text'])
    sanders += word_in_text('sanders', row['text'])
    cruz += word_in_text('cruz', row['text'])

# Plot twitter data
import seaborn as sns
import matplotlib.pyplot as plt
# Set seaborn style
sns.set(color_codes=True)
# Create a list of labels:cd
cd = ['clinton', 'trump', 'sanders', 'cruz']
# Plot histogram
ax = sns.barplot(cd, [clinton, trump, sanders, cruz])
ax.set(ylabel="count")
plt.show()