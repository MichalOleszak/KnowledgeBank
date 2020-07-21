# Capture IPython Shell output --------------------------------------------------------------------

# Save output of shell's ls in a python variable called output
%%bash --out output
ls

# Get size of files in a directory
!ls -l | awk '{SUM+=$5} END {print SUM}'


# Type of test, where test = !some_shell_command, is an SList
# SLits are the special type of data captured from shell commnads in iPython
# They have 3 main methods:
#  - fields
#  - grep
#. - sort

# Fields used for selecting whitespace-separated columns:
ls = !ls -l /usr/bin
ls.fields(1, 5)[1:4] 

# Grep used for looking for utilities that kill unix processes
ls.grep("kill")

#`Sort used for sorting disk-using processes by numerical values
disk_usage = !df -h
# sort by usage
disk_usage.sort(5, nums=True)


# Execute shell commands in subprocess ------------------------------------------------------------
import subprocess

# Run the `ps aux command that lists running processes
with subprocess.Popen(["ps", "aux"], stdout=subprocess.PIPE) as proc:
    process_output_iterator = proc.stdout.readlines()
    process_output_string = proc.stdout.read()

# Find installed packages
with Popen(["pip","list","--format=json"], stdout=PIPE) as proc:
  result = proc.stdout.read()
converted_result = json.loads(result)
pprint.pprint(converted_result)

# Waiting for long-running processes and kill them if you wait too long
proc = Popen(["sleep", "6"], stdout=PIPE, stderr=PIPE)
try:
    output, error = proc.communicate(timeout=5)
except TimeoutExpired:
    proc.kill()

# Detecting duplicate files
checksums = {}
duplicates = []
for filename in files:
	#  md5sum utility is a shell command that finds the unique hash of each file
    with Popen(["md5sum", filename], stdout=PIPE) as proc:
        checksum, _ = proc.stdout.read().split()
        if checksum in checksums:
            duplicates.append(filename)
        checksums[checksum] = filename


# Sending input to processes ----------------------------------------------------------------------

# Using Unix pipes
proc1 = Popen(["process_one.sh"], stdout=subprocess.PIPE)
Popen(["process_two.sh"], stdin=proc1.stdout)

# Using run method, a higher level abstraction (recommended)
proc1 = run(["process_one.sh"], stdout=subprocess.PIPE)
run(["process_two.sh"], stdin=proc1.stdout)

# Passing arguments safely to shell commands
# By edfalut, in Popen and run, shell=False -> this is safe and recommended
# shell=True allows running arbitary code, e.g. malicious software
# If shell=True is really needed, shelx should be used to construct shell commands safely
# Example: want to calculate disk usage in user-provided files; make sure user won't pass 
# malicious code as their input that would e.g. delete the files
user_input = "pluto mars jupiter"
sanitized_user_input = shlex.split(user_input)
# Safely Extend the command with sanitized input
cmd = ["du", "-sh", "--total"]
cmd.extend(sanitized_user_input)
disk_total = subprocess.run(cmd, stdout=subprocess.PIPE)


# Walking the filesystem --------------------------------------------------------------------------

# Find all .csv files
matches = []
for root, _, files in os.walk('test_dir'):
    for name in files:
        fullpath = os.path.join(root, name)
        print(f"Processing file: {fullpath}")
        _, ext = os.path.splitext(fullpath)
        if ext == ".csv":
            matches.append(fullpath)

# Find files matching a pattern
from pathlib import path
p = Path("data")
list(p.glob("*.csv"))
# Recursive search
list(p.glob("**/*.csv"))

# Unix check for the pattern's presence, returns True/False
fnmatch.fnmatch(file, "*.csv")

# Converting fnmatch to regular expressions
import fnmatch, re
regex = fnmatch.translate("*.csv")
pattern = re.compile(regex)
pattern.match("myfile.csv")

# High-level file and directory operations
# - shutil allows to copy, delete or archive the tree
# - tempfile generates temporary files and dirs
from shutil import copytree, ignore_patterns, rmtree, make_archive
# Copy entire directory tree but the text and csv files
copytree(source, destination, ignore=ignore_patterns("*.txt", "*.csv"))
# Remove a tree
rmtree(source, destination)
# Archive a tree
make_archive("somearchive", "gztar", "destination")

# Create a self-destructing temporary file
with tempfile.NamedTemporaryFile() as exploding_file:
  	# This file will be deleted automatically after the with statement block
    print(f"Temp file created: {exploding_file.name}")
    exploding_file.write(b"This message will self-destruct in 5....4...\n")


# Decorators --------------------------------------------------------------------------------------

# A decorator that will wrap a function and print any *args or **kw arguments out.
# create decorator
def debug(f):
	@wraps(f)
	def wrap(*args, **kw):
		result = f(*args, **kw)
		print(f"function name: {f.__name__}, args: [{args}], kwargs: [{kw}]")
		return result
	return wrap
# apply decorator
@debug
def mult(x, y=10):
	return x*y
print(mult(5, y=5))


# Click to automate writing CLTs ------------------------------------------------------------------
# Click: Python package to write beautiful command line interfaces
# - arbitrary nesting of commands
# - automatic help page generation
# - lazy loading of subcommands at runtime
import click
@ click.command()
@ click.option("--phrase", prompt="Enter a phrase", help="")
def tokenize(phrase):
	click.echo(f"Tokenized phrase: {phrase.split()}")
if __name__ == "__main__":
	tokenize()

# Create a CLI with subcommands:
@click.group()
def cli(): 
	pass
@cli.command()
def one():
	click.echo("One")
@cli.command()
def two():
	click.echo("Two")
if __name__ == "__main__":
	cli()
# Display commands and arguemnts available
python test.py
# Run specifig functions
python test.py one

# Writing and reading files with click
words = ["Asset", "Bubble", "10", "Year"]
filename = "words.txt"
with click.open_file(filename, 'w') as f:
    for word in words:
        f.write(f'{word}\n')
with open(filename) as output_file:
    print(output_file.read())

# Invoking command line tests
@click.command()
@click.option("--num", default=2, help="Number of clusters")
def run_cluster(num):
    result = myfunction(num)
    click.echo(f'Cluster assignments: {result} for total clusters [{num}]')
# Create the click test runner
runner = CliRunner()
# Run the click app and assert it runs without error
result = runner.invoke(run_cluster, ['--num', '2'])
assert result.exit_code == 0
print(result.output)
