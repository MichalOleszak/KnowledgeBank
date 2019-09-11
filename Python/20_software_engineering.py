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


# Docstrings ----------------------------------------------------------------------------------------------------------
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

# Google docstring style
"""Google style.

The Google style tends to result in
wider docstrings with fewer lines of code.

Section 1:    
	Item 1: Item descriptions don't need line breaks.
"""

# Numpy docstring style
"""Numpy style.

The Numpy style tends to results in
narrower docstrings with more lines of code.

Section 1
---------
Item 1    
	Item descriptions are indented on a new line.
"""

# Building docstring from multiple strings in parenthesis
def get_matches(word_list: List[str], query:str) -> List[str]:
    ("Find lines containing the query string.\nExamples:\n\t"
     # Complete the docstring example below
     ">>> get_matches(['a', 'list', 'of', 'words'], 's')\n\t"
     # Fill in the expected result of the function call
     "['list', 'words']")
    return [line for line in word_list if query in line]
	

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

# Parametrizing test
@pytest.mark.parametrize("inputs", ["intro.md", "plot.py", "discussion.md"])
def test_nbuild(inputs):
    assert nbuild([inputs]).cells[0].source == Path(inputs).read_text()

# Check whether error is raised
@pytest.mark.parametrize("not_exporters", ["htm", "ipython", "markup"])
def test_nbconv(not_exporters):
    with pytest.raises(ValueError):
        nbconv(nb_name="mynotebook.ipynb", exporter=not_exporters)
	

# Classmethods --------------------------------------------------------------------------------------------------------
# Due to the classmethod decorator, the first argument in the function it decorates is not the class' instance (self),
# but the class itself. This allows to instatiate multiple instances at ones.

class TextFile:
	
	instances = []
	
	def__init__(self, file):        
		self.text = Path(file).read_text()        
		self.__class__.instances.append(file)    
		
	@classmethod
	def instantiate(cls, filenames):
		return map(cls, filenames)
		
boston, diabetes = TextFile.instantiate(['boston.txt', 'diabetes.txt'])
TextFile.instances


# Type annotations ----------------------------------------------------------------------------------------------------
from typing import List, Optional

# Example 1
class TextFile:
    def __init__(self, name: str) -> None:
        self.text = Path(name).read_text()
	# Return list of strings
    def get_lines(self) -> List[str]:
        return self.text.split("\n")

# Example 2
class MatchFinder:
	# MatchFinder should only accept a list of strings as its strings argument
    def __init__(self, strings: List[str]) -> None:
        self.strings = strings
	# The get_matches() method returns a list of either
	# 	- every string in strings that contains the query argument or
    # 	- all strings in strings if the match argument is None.
    def get_matches(self, query: Optional[str] = None) -> List[str]:
        return [s for s in self.strings if query in s] if query else self.strings

		
# Building Jupyter notebooks from within python code ------------------------------------------------------------------
from nbformat.v4 import new_notebook, new_code_cell
from nbconvert.exporters import get_exporter
from pathlib import Path


def nbuild(filenames: List[str]) -> nbformat.notebooknode.NotebookNode:
    """Create a Jupyter notebook from text files and Python scripts."""
    nb = new_notebook()
    nb.cells = [
        # Create new code cells from files that end in .py
        new_code_cell(Path(name).read_text()) 
        if name.endswith(".py")
        # Create new markdown cells from all other files
        else new_markdown_cell(Path(name).read_text()) 
        for name in filenames
    ]
    return nb
	
	
def nbconv(nb_name: str, exporter: str = "script") -> str:
    """Convert a notebook into various formats using different exporters."""
    # Instantiate the specified exporter class
    exp = get_exporter(exporter)()
    # Return the converted file"s contents string 
    return exp.from_filename(nb_name)[0]


# Shell Command Line Interfaces (CLIs) --------------------------------------------------------------------------------

# With argparse

def argparse_cli(func: Callable) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("in_files", nargs="*")
    args = parser.parse_args()
    print(func(args.in_files))

if __name__ == "__main__":
    argparse_cli(nbuild)

# With docopt
# In docopt docstrings, optional arguments are wrapped in square brackets ([]), 
# while lists of arguments are followed by ellipses (...).

"""Usage: docopt_cli.py [IN_FILES...]"""
def docopt_cli(func: Callable) -> None:
    # Assign the shell arguments to "args"
    args = docopt(__doc__)
    print(func(args["IN_FILES"]))

if __name__ == "__main__":
    docopt_cli(nbuild)


# GitPython -----------------------------------------------------------------------------------------------------------

# Initialize a new repo in the current folder
repo = git.Repo.init()

# Obtain a list of untracked files
untracked = repo.untracked_files

# Add all untracked files to the index
repo.index.add(untracked)

# Commit newly added files to version control history
repo.index.commit(f"Added {', '.join(untracked)}")
print(repo.head.commit.message)

# Commit modified files
changed_files = [file.b_path
                 # Iterate over items in the diff object
                 for file in repo.index.diff(None)
                 # Include only modified files
                 .iter_change_type("M")]

repo.index.add(changed_files)
repo.index.commit(f"Modified {', '.join(changed_files)}")
for number, commit in enumerate(repo.iter_commits()):
    print(number, commit.message)


# Virtual environments ------------------------------------------------------------------------------------------------

# Create a venv and check pandas version
venv.create(".venv")
cp = subprocess.run([".venv/bin/python", "-m", "pip", "list"], stdout=-1)
for line in cp.stdout.decode().split("\n"):
    if "pandas" in line:
        print(line)

# Install a package and show info
print(run([".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"], stdout=-1).stdout.decode())
print(run([".venv/bin/python", "-m", "pip", "show", "aardvark"], stdout=-1).stdout.decode())
