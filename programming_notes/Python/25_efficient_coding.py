# Measuring runtime
%timeit mylist = list()
%timeit mylist = []

# Multiple lines
%%timeit
a = 5
b = a * 5

# Profiling for runtime
import line_profiler
%load_ext line_profiler
%lprun -f my_func my_func(a, b)

# Profiling for memory footprint
import memory_profiler
%load_ext memory_profiler
from file.py import my_func
%mprun -f my_func my_func(a, b)

# Counting
from collection import Counter
Counter({'a', 'b', 'c', 'c', 'a', 'a'])

# Combining in a list of tuples
[*zip(list_1, list_2, list_3)]

# Getting all possible n-element subsets
from itertools import combinations
[*combinations(mylist, n)]

# Looping over pandas dataframe: itertuples is faster than iterrows
# row is a namedtuple object; it only supports the dot reference
for row in df.itertuples():
  i = row.Index
  var = row.var
  
  