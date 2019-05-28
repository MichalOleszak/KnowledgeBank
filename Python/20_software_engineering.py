# PEP8 -----------------------------------------------------------------------------------------------------------------
import pycodestyle

# Create a StyleGuide instance
style_checker = pycodestyle.StyleGuide()
# Run PEP 8 check on multiple files
result = style_checker.check_files(['nay_pep8.py', 'yay_pep8.py'])
# Print result of PEP 8 style check
print(result.messages)

# Installing requirements from file
pip install -r requirements.txt


# Classes --------------------------------------------------------------------------------------------------------------
# working_dir
# ├── text_analyzer
# │    ├── __init__.py
# │    ├── counter_utils.py
# │    ├── document.py
# └── my_script.py

# Creating the class
# working in document.py

# Define Document class
# When creating the instance of the class, __init__ will be run automatically. Non-punlic methods, starting witg "_",
# are for internal use; users can use them, but do so at their own risk (no documentation, unannounced changes, etc.)

class Document:
    """A class for text analysis
    
    :param text: string of text to be analyzed
    :ivar text: string of text to be analyzed; set by `text` parameter
    """
    # Method to create a new instance of MyClass
    def __init__(self, text):
        # Store text parameter to the text attribute
        self.text = text
        # Tokenize the document with non-public tokenize method
    	self.tokens = self._tokenize()
    	# Perform word count with non-public count_words method
    	self.word_counts = self._count_words

    # Assumes tokenize and Counter from other packages are imported
    def _tokenize(self):
    	return tokenize(self.text)
	
  	# Non-public method to tally document's word counts with Counter
  	def _count_words(self):
    	return Counter(self.tokens)

# Can be imported with
from .document import Document

# Using the  functionality
# Import custom text_analyzer package
import text_analyzer
# Create an instance of Document with datacamp_tweet
my_document = text_analyzer.Document(text=datacamp_tweet)
# Print the text attribute of the Document instance
print(my_document.text)
# Print the first 5 tokens from datacamp_doc
print(datacamp_doc.tokens[:5])
# Print the top 5 most used words in datacamp_doc
print(datacamp_doc.word_counts.most_common(5))


# Class inheritance ----------------------------------------------------------------------------------------------------
# Child inherits all attributes from the parant, plus can have some extensions

# Define a SocialMedia class that is a child of the `Document class`
class SocialMedia(Document):
    def __init__(self, text):
        Document.__init__(self, text)
        self.hashtag_counts = self._count_hashtags()
        self.mention_counts = self._count_mentions()
        
    def _count_hashtags(self):
        # Filter attribute so only words starting with '#' remain
        return filter_word_counts(self.word_counts, first_char='#')      
    
    def _count_mentions(self):
        # Filter attribute so only words starting with '@' remain
        return filter_word_counts(self.word_counts, first_char='@')

# Using child class
# Import custom text_analyzer package
import text_analyzer
# Create a SocialMedia instance with datacamp_tweets
dc_tweets = text_analyzer.SocialMedia(text=datacamp_tweets)
# Print the top five most most mentioned users
print(dc_tweets.mention_counts.most_common(5))
# Plot the most used hashtags
text_analyzer.plot_counter(dc_tweets.hashtag_counts)


# Multilevel inheritance -----------------------------------------------------------------------------------------------
# One child can inherit from multiple parents

# Define a Tweet class that inherits from SocialMedia
class Tweets(SocialMedia):
    def __init__(self, text):
        # Call parent's __init__ with super()
        super().__init__(self, text)
        # Define retweets attribute with non-public method
        self.retweets = self._process_retweets()

    def _process_retweets(self):
        # Filter tweet text to only include retweets
        retweet_text = filter_lines(self.text, first_chars='RT')
        # Return retweet_text as a SocialMedia object
        return SocialMedia(retweet_text)

# Using the grandchild
# Import needed package
import text_analyzer
# Create instance of Tweets
my_tweets = text_analyzer.Tweets(datacamp_tweets)
# Plot the most used hashtags in the tweets
my_tweets.plot_counts('hashtag_counts')
# Plot the most used hashtags in the retweets
my_tweets.retweets.plot_counts('hashtag_counts')


# Proper docstring example --------------------------------------------------------------------------------------------
def tokenize(text, regex=r'[a-zA-z]+'):
  """Split text into tokens using a regular expression

  :param text: text to be tokenized
  :param regex: regular expression used to match tokens using re.findall 
  :return: a list of resulting tokens

  >>> tokenize('the rain in spain')
  ['the', 'rain', 'in', 'spain']
  """
  return re.findall(regex, text, flags=re.IGNORECASE)

# Print the docstring
help(tokenize)


# The Zen of Python ---------------------------------------------------------------------------------------------------
import this


# Testing with doctest ------------------------------------------------------------------------------------------------
def sum_counters(counters):
    """Aggregate collections.Counter objects by summing counts

    :param counters: list/tuple of counters to sum
    :return: aggregated counters with counts summed

    >>> d1 = text_analyzer.Document('1 2 fizz 4 buzz fizz 7 8')
    >>> d2 = text_analyzer.Document('fizz buzz 11 fizz 13 14')
    >>> sum_counters([d1.word_counts, d2.word_counts])
    Counter({'buzz': 2, 'fizz': 4})
    """
    return sum(counters, Counter())

doctest.testmod()


# Testing with pytest -------------------------------------------------------------------------------------------------
# working_dir
# ├── text_analyzer
# │    ├── __init__.py
# │    ├── counter_utils.py
# │    ├── document.py
# ├──  setup.py
# ├──  requirements.py
# └── tests
#      └── test_unit.py
from collections import Counter
from text_analyzer import SocialMedia

# Create an instance of SocialMedia for testing
test_post = 'learning #python & #rstats is awesome! thanks @datacamp!'
sm_post = SocialMedia(test_post)

# Test hashtag counts are created properly
def test_social_media_hashtags():
    expected_hashtag_counts = Counter({'#python': 1, '#rstats': 1})
    assert sm_post.hashtag_counts == expected_hashtag_counts

# Run from command line
$ pytest



