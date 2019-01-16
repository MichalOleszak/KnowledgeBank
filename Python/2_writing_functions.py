# Tuples (similar to lists, but immutable = cannot change values) ------------
# used to return multiple values from function
def myfun():
	"""Docstring."""
	even_nums = (2, 4, 6)
	return(even_nums)
a, b, c = myfun()         # set a = 2, b = 4, c = 6


# Scope and user-defined functions -------------------------------------------
num = 5
def func1():
    num = 3
    print(num)
def func2():
    global num
    double_num = num * 2
    num = 6
    print(double_num)
func1()                   # prints 3
func2()                   # prints 10
num                       # num is now 6


# Nested functions -----------------------------------------------------------
## Return function as functions output
def raise_val(n):
	"""Return the inner function."""
	def inner(x):
		"""Raise x to the power of n."""
		raised = x ** n
		return raised
	return inner
	
## Use the keyword nonlocal within a nested function to alter the value 
## of a variable defined in the enclosing scope.
def outer():
	"""Prints the value of n."""
	n = 1
	def inner():
		nonlocal n
		n = 2
		print(n) 
	inner()
	print(n)
outer()                   # prints 2

## Closure: 
## The nested or inner function remembers the state of its enclosing scope when called. 
## Thus, anything defined locally in the enclosing scope is available to the inner function 
## even when the outer function has finished execution.
def echo(n):
    """Return the inner_echo function."""
    def inner_echo(word1):
        """Concatenate n copies of word1."""
        echo_word = word1 * n
        return echo_word
    return(inner_echo)

twice = echo(2)
twice('hello')            # prints "hellohello"


# Function with variable-length arguments (*args) ----------------------------
# (within the function definition, args is a tuple)
def gibberish(*args):
    """Concatenate strings in *args together."""
    hodgepodge = str()
    for word in args:
        hodgepodge += word
    return(hodgepodge)
	
	
# Function with variable-length keyword arguments (**kwargs) -----------------
# (within the function definition, kwargs is a dictionary)
def report_status(**kwargs):
    """Print out the status of a movie character."""
    # Iterate over the key-value pairs of kwargs
    for key, value in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(key + ": " + value)
report_status(name="luke", affiliation="jedi", status="missing")


# Lambda functions -----------------------------------------------------------
sum = (lambda x, y: x + y)

## Map & lambda funs
spells = ["protego", "accio", "expecto patronum", "legilimens"]
shout_spells = map(lambda item: item + '!!!', spells)
shout_spells_list = list(shout_spells)

## Filter & lambda funs
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
result = filter(lambda member: len(member) > 6, fellowship)
result_list = list(result)

## Reduce & lambda funs
from functools import reduce 
stark = ['robb', 'sansa', 'arya', 'eddard', 'jon']
result = reduce(lambda item1, item2: item1 + item2, stark)


# Error handling -------------------------------------------------------------

## Error handling with try-except 
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""
    echo_word = str()
    shout_words = str()
    try:
        echo_word = word1 * echo
        shout_words = echo_word + '!!!'
    except:
        print("word1 must be a string and echo must be an integer.")
    return shout_words
	
shout_echo("particle", echo="accelerator")

## Error handling by raising an error
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""
    if echo < 0:
        raise ValueError('echo must be greater than 0')
    echo_word = word1 * echo
    shout_word = echo_word + '!!!'
    return shout_word
	
shout_echo("particle", echo=-2)