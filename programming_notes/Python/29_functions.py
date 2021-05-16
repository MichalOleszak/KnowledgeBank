# Context managers ------------------------------------------------------------

# Timer
# Add a decorator that will make timer() a context manager
@contextlib.contextmanager
def timer():
  """Time the execution of a context block.

  Yields:
    None
  """
  start = time.time()
  # Send control back to the context block
  yield
  end = time.time()
  print('Elapsed: {:.2f}s'.format(end - start))

with timer():
  print('This should take approximately 0.25 seconds')
  time.sleep(0.25)


# Read only
@contextlib.contextmanager
def open_read_only(filename):
  """Open a file in read-only mode.

  Args:
    filename (str): The location of the file to read

  Yields:
    file object
  """
  read_only_file = open(filename, mode='r')
  # Yield read_only_file so it can be assigned to my_file
  yield read_only_file
  # Close read_only_file
  read_only_file.close()

with open_read_only('my_file.txt') as my_file:
  print(my_file.read())


# Scopes (where Python looks for object) ------------------------------------
# 1. local = thefunction
# 2. nonlocal = partent function (for nested funcs)
# 3. global = outside of the function
# 4. builtin = ex. print() function


# Closures ------------------------------------------------------------------
# A tuple of variables no logner in scope, but that a function needs.

def foo():
  a = 1
  def bar():
    print(a)
  return bar

f = foo()
f()  #  ---> prints 1

# f knows the value of a defined in the scope of another function, because python
# attached it to the clouse of bar:

f.__closure__[0].cell_contents  #  ---> 1


# Decorators -----------------------------------------------------------------
# - Wrapper around the function that changes its behaviour
# - Take a func as argument and return its modified version

def double_args(f):
  def wrapper(a, b):
    return f(a * 2, b * 2)
  return wrapper


@double_args
def multiply(a, b):
  return a * b

multiply(2, 3)  #  ---> returns 24

# instead of decorating multiply, we could do:
double_args(multiply)(2, 3)
# or equivalently
multiply = double_args(multiply)
multiply(2, 3)


# ------

def print_before_and_after(func):
  def wrapper(*args):
    print('Before {}'.format(func.__name__))
    # Call the function being decorated with *args
    func(*args)
    print('After {}'.format(func.__name__))
  # Return the nested function
  return wrapper

@print_before_and_after
def multiply(a, b):
  print(a * b)

multiply(5, 10)

# ----

def timer(func):
  """A decorator that prints how long a function took to run."""
  def wrapper(*args, **kwargs):
    t_start = time.time()
    result = func(*args, **kwargs)   
    t_total = time.time() - t_start    
    print('{} took {}s'.format(func.__name__, t_total))
    return result
  return wrapper

# ----

def memoize(func):
"""Store the results of the decorated function for fast lookup  """
  cache = {}
  def wrapper(*args, **kwargs):
    if (args, kwargs) not in cache:
      cache[(args, kwargs)] = func(*args, **kwargs)
    return cache[(args, kwargs)]
  return wrapper

# ---

def print_return_type(func):
  def wrapper(*args, **kwargs):
    result = func(*args, **kwargs)
    print('{}() returned type {}'.format(
      func.__name__, type(result)
    ))
    return result
  return wrapper
  
@print_return_type
def foo(value):
  return value
  
print(foo(42))
print(foo([1, 2, 3]))
print(foo({'a': 42}))

# ---

def counter(func):
  def wrapper(*args, **kwargs):
    wrapper.count += 1
    return func(*args, **kwargs)
  wrapper.count = 0
  return wrapper

@counter
def foo():
  print('calling foo()')
  
foo()
foo()

print('foo() was called {} times.'.format(foo.count))

# Preserving docstrings when decorating functions
# Below, we are printing wrapper()'s docstring, which is empty.

def add_hello(func):
  def wrapper(*args, **kwargs):
    print('Hello')
    return func(*args, **kwargs)
  return wrapper

@add_hello
def print_sum(a, b):
  """Adds two numbers and prints the sum"""
  print(a + b)
  
print_sum(10, 20)
print(print_sum.__doc__)

# Fix:
from functools import wraps

def add_hello(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    print('Hello')
    return func(*args, **kwargs)
  return wrapper


# Decorators that take arguments
# - Decorators can only take 1 arg: the func they decorate
# - Solution: create a function that returns a decorator (decorator factory)
# - Decorate one's func with the result of calling the decorator factort = use ()!

def run_n_times(n):
  """Define and return a decorator"""
  def decorator(func):
    def wrapper(*args, **kwargs):
      for i in range(n):
        func(*args, **kwargs)
    return wrapper
  return decorator

@run_n_times(10)
def print_sum(a, b):
  print(a + b)
  
print_sum(15, 20)

# ----

def timeout(n_seconds):
  def decorator(func):    
    @wraps(func)
    def wrapper(*args, **kwargs):
      signal.alarm(n_seconds)
      try:
        return func(*args, **kwargs)
      finally:
        signal.alarm(0)
    return wrapper
  return decorator

# ----

def tag(*tags):
  # Define a new decorator, named "decorator", to return
  def decorator(func):
    # Ensure the decorated function keeps its metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
      # Call the function being decorated and return the result
      return func(*args, **kwargs)
    wrapper.tags = tags
    return wrapper
  # Return the new decorator
  return decorator

@tag('test', 'this is a tag')
def foo():
  pass

print(foo.tags)



