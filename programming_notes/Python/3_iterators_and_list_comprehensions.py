# Iterators & iterables ------------------------------------------------------
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']
my_iterator = iter(flash)
next(my_iterator)
next(my_iterator)


# Enumerate ------------------------------------------------------------------
# enumerate() returns an enumerate object that produces a sequence of tuples (= index-value pairs)
# Create a list of tuples
flash_list = list(enumerate(flash))
# Unpack and print the tuple pairs
for index1, value1 in enumerate(flash):
    print(index1, value1)
# Change the start index
for index2, value2 in enumerate(flash, start=1):
    print(index2, value2)


# Zip ------------------------------------------------------------------------
# zip() takes any number of iterables and returns a zip object that is an iterator of tuples

mutants = ['charles xavier',
           'bobby drake',
           'kurt wagner',
           'max eisenhardt',
           'kitty pride']
powers = ['telepathy',
          'thermokinesis',
          'teleportation',
          'magnetokinesis',
          'intangibility']
# Create a list of tuples
mutant_data = list(zip(mutants, powers))
# Create a zip object
mutant_zip = zip(mutants, powers)
# Unpack the zip object and print the tuple values
for value1, value2 in mutant_zip:
    print(value1, value2)


# Unzipping ------------------------------------------------------------------
# * unpacks an iterable such as a list or a tuple into positional arguments in a function call
mutant_zip = zip(mutants, powers)
print(mutant_zip)
print(*mutant_zip)

z1 = zip(mutants, powers)
result1, result2 = zip(*z1)
# Check if unpacked tuples are equivalent to original tuples
print(result1 == mutants)
print(result2 == powers)


# Loading data in chunks -----------------------------------------------------
import pandas as pd


def count_entries(csv_file, c_size, colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    counts_dict = {}
    for chunk in pd.read_csv(csv_file, chunksize=c_size):
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1
    return counts_dict


print(count_entries("data/tweets.csv", 10, "lang"))


# Loading data line by line using generator functions ------------------------
def read_large_file(file_object):
    """A generator function to read a large file lazily."""
    # Loop indefinitely until the end of the file
    while True:
        # Read a line from the file: data
        data = file_object.readline()
        # Break if this is the end of the file
        if not data:
            break
        yield(data)

    with open('data/tweets.csv') as file:
        # Create a generator object for the file: gen_file
        gen_file = read_large_file(file)
        # Print the first three lines of the file
        print(next(gen_file))
        print(next(gen_file))
        print(next(gen_file))


# List comprehensions --------------------------------------------------------
# List comprehensions collapse for loops for building lists into a single line
squares = [i**2 for i in range(10)]

# Nested list comprehensions
matrix = [[col for col in range(5)] for row in range(5)]
for row in matrix:
    print(row)

# Conditionals in comprehensions
 cond_on_iterable = [num ** 2 for num in range(10) if num % 2 == 0]
 cond_on_output =  [num ** 2 if num % 2 == 0 else 0 for num in range(10)]

# Dict comprehensions
# the main difference between a list comprehension and a dict comprehension is the use of curly braces {}
# instead of []. Additionally, members of the dictionary are created using a colon :, as in key:value.
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
dict_comp = {member:len(member) for member in fellowship}


# Generator expressions ------------------------------------------------------
# like list comprehensions, but do not actually create the list (no need to store in memory),
# but allow to iterate over its elements and create them when needed (lazy evaluation)
squares = (i**2 for i in range(10))
print(squares)
for i in squares:
    print(i)


# Generator function ---------------------------------------------------------
# use "yield" instead of "return"
def num_sequence(n):
 """Generate values from 0 to n."""
 i = 0
 while i < n:
 yield i
 i += 1

 def get_lengths(input_list):
     """Generator function that yields the
     length of the strings in input_list."""
     for person in input_list:
         yield (len(person))

 for value in get_lengths(['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']):
     print(value)