# Connecting to a database ---------------------------------------------------------------------------------------------
from sqlalchemy import create_engine
engine = create_engine('sqlite:///census.sqlite')
# Print table names
engine.table_names()

# Autoloading Tables from a Database
# SQLAlchemy can be used to automatically load tables from a database using something called reflection.
# Reflection is the process of reading the database and building the metadata based on that information.
# It's the opposite of creating a Table by hand.
from sqlalchemy import MetaData, Table
# Get metadata
metadata = MetaData()
# Reflect census table from the engine
census = Table('census', metadata, autoload=True, autoload_with=engine)
# Print census table metadata
print(repr(census))

# Viewing Table Details
# Print the column names
print(census.columns.keys())
# Print full table metadata
print(repr(metadata.tables['census']))

# Selecting data from a Table: raw SQL
connection = engine.connect()
query = 'SELECT * FROM census'
results = connection.execute(query).fetchall()

# Selecting data from a Table with SQLAlchemy
from sqlalchemy import select
census = Table('census', metadata, autoload=True, autoload_with=engine)
query = select([census])
print(query)    # prints the SQL query: SELECT * FROM census
results = connection.execute(query).fetchall()

# The separation between the ResultSet and ResultProxy allows us to fetch as much or as little data as we desire.
# -> ResultProxy: The object returned by the .execute() method. It can be used in a variety of ways to get the data
#    returned by the query.
# -> ResultSet: The actual data asked for in the query when using a fetch method such as .fetchall() on a ResultProxy.
first_row = results[0]
print(first_row[0])          # Print the first column of the first row by using an index
print(first_row['state'])    # Print the 'state' column of the first row by using its name


# Filtering and Targeting Data -----------------------------------------------------------------------------------------
# Filter on one column value
query = select([census]).where(census.columns.state == 'New York')
results = connection.execute(query).fetchall()

# Expressions: in_(), not_(), or_(), and_(), ...
states = ['New York', 'California', 'Texas']
query = select([census]).where(census.columns.state.in_(states))

select([census]).where(
  and_(census.columns.state == 'New York',
       or_(census.columns.age == 21,
          census.columns.age == 37
         )
      )
  )

# Ordering query results
# by a single columns
query = select([census.columns.state]).order_by(census.columns.state)           # ascending order by default
query = select([census.columns.state]).order_by(desc(census.columns.state))     # descending order
# by multiple columns
query = select([census.columns.state, census.columns.age]).order_by(census.columns.state, desc(census.columns.age))
results = connection.execute(query).fetchall()
print(results[:20])

# Counting, Summing and Grouping Data
from sqlalchemy import func
# sum and count
select([func.sum(census.columns.pop2008)])
select([func.count(census.columns.pop2008)])
# number of distinct values
stmt = select([func.count(census.columns.state.distinct())])
distinct_state_count = connection.execute(stmt).scalar()
print(distinct_state_count)
# get a count for each record with a particular value in another column
# count of records (in age column) by state
select([census.columns.state, func.count(census.columns.age)]).group_by(census.columns.state)
# Population Sum by State
pop2008_sum = func.sum(census.columns.pop2008).label('population')    # Get sum of pop2008 labeled as population
select([census.columns.state, pop2008_sum]).group_by(census.columns.state)

# SQLAlchemy & Pandas
results = connection.execute(query).fetchall()
df = pd.DataFrame(results)
df.columns = results[0].keys()


# Advanced SQLAlchemy Queries ------------------------------------------------------------------------------------------
# Calculating Values in a Query: difference between two oclumns
query = select([census.columns.state, (census.columns.pop2008-census.columns.pop2000).label('pop_change')])
# Calculating Values in a Query: overall percentage of females
from sqlalchemy import case, cast, Float
female_pop2000 = func.sum(
    case([
        (census.columns.sex == 'F', census.columns.pop2000)
    ], else_=0))
total_pop2000 = cast(func.sum(census.columns.pop2000), Float)
query = select([female_pop2000 / total_pop2000 * 100])

# SQL Relationships
# Automatic Joins with an Established Relationship
# If you have two tables that already have an established relationship, you can automatically use that relationship by
# just adding the columns we want from each table to the select statement
select([census.columns.pop2008, state_fact.columns.abbreviation])
# Joins
# If you aren't selecting columns from both tables or the two tables don't have a defined relationship,
# you can still use the .join() method on a table to join it with another table
select([census, state_fact]).select_from(
    census.join(state_fact, census.columns.state == state_fact.columns.name))

# Working with Hierarchical Tables
# Example: employess table: id, name, job, manager; manager is also an employee, has his own record
# alias() allows to refer to the same table with two unique names
managers = employees.alias()
query = select(
    [managers.columns.name.label('manager'),
     employees.columns.name.label('employee')])
query.select_from(
    employees.join(managers, managers.columns.id == employees.colunms.manager))

# Dealing with Large ResultSets
# Start a while loop checking for more results
while more_results:
    # Fetch the first 50 results from the ResultProxy: partial_results
    partial_results = results_proxy.fetchmany(50)
    # if empty list, set more_results to False
    if partial_results == []:
        more_results = False
    # Loop over the fetched records and increment the count for the state
    for row in partial_results:
        state_count[row.state] += 1

results_proxy.close()


# Creating Databases and Tables ----------------------------------------------------------------------------------------
# Create a table
from sqlalchemy import Table, Column, String, Integer, Float, Boolean
data = Table('data', metadata,
             Column('name', String(255)),
             Column('count', Integer()),
             Column('amount', Float()),
             Column('valid', Boolean())
)
metadata.create_all(engine)
print(repr(data))

# Constraints and Data Defaults
data = Table('data', metadata,
             Column('name', String(255), unique=True),
             Column('count', Integer(), default=1),
             Column('amount', Float()),
             Column('valid', Boolean(), default=False)
)

# Inserting Data into a Table
# Inserting a single row
from sqlalchemy import insert, select
stmt = insert(data).values(name='Anna', count=1, amount=1000.00, valid=True)
results = connection.execute(stmt)
# Build a select statement to validate the insert
stmt = select([data]).where(data.columns.name == 'Anna')
print(connection.execute(stmt).first())
# Inserting Multiple Records at Once
values_list = [
    {'name': 'Anna', 'count': 1, 'amount': 1000.00, 'valid': True},
    {'name': 'Taylor', 'count': 1, 'amount': 750.00, 'valid': False}
]
stmt = insert(data)
results = connection.execute(stmt, values_list)

# Loading a CSV into a Table
stmt = insert(census)
values_list = []
total_rowcount = 0

# Enumerate the rows of csv_reader
for idx, row in enumerate(csv_reader):
    #create data and append to values_list
    data = {'state': row[0], 'sex': row[1], 'age': row[2], 'pop2000': row[3],
            'pop2008': row[4]}
    values_list.append(data)
    # Check to see if divisible by 51
    if idx % 51 == 0:
        results = connection.execute(stmt, values_list)
        total_rowcount += results.rowcount
        values_list = []

# Updating Data in a Database
# Updating individual records
from sqlalchemy import update
stmt = update(employees).where(employees.columns.id == 3).values(active=True)
result_proxy = connection.execute(stmt)
# Updating multiple rows
# example 1: set all avtive employess to not active with salary 0
stmt = update(employees.where(
             employees.columns.active == True)
).values(active=False, salary=0.00)
result_proxy = connection.execute(stmt)
# example 2: update the value of notes column to be "The Wild West" if region name is "West"
stmt = update(state_fact).values(notes="The Wild West").where(state_fact.columns.census_region_name == 'West')
results = connection.execute(stmt)
# Correlated updates
# You can also update records with data from a select statement. This is called a correlated update.
# It works by defining a select statement that returns the value you want to update the record with
# and assigning that as the value in an update statement.
# Give everyone the maximum salary
new_salary = select([employees.columns.salary])
new_salary = new_salary.order_by(desc(employees.columns.salary))
new_salary = new_salary.limit(1)    # get the maximum salary
stmt = update(employees).values(salary=new_salary)
result_proxy = connection.execute(stmt)

# Removing Data From a Database
# Deleting all the records from a table
from sqlalchemy import delete, select
stmt = delete(census)
results = connection.execute(stmt)
# Print affected rowcount
print(results.rowcount)
# Verify there is no rows
stmt = select([census])
print(connection.execute(stmt).fetchall())

# Deleting specific records
delete(employees).where(employees.columns.id == 3)
delete(census).where(and_(census.columns.sex == 'M', census.columns.age == 36))

# Deleting a Table Completely
# Drop the state_fact table
state_fact.drop(engine)
# Check to see if state_fact exists
print(state_fact.exists(engine))
# Drop all tables
metadata.drop_all(engine)
# Check to see if census exists
print(census.exists(engine))


# Full process example - a case study ----------------------------------------------------------------------------------
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, insert, select, case, cast

# 1. Setup the Engine and MetaData
engine = create_engine('sqlite:///chapter5.sqlite')
metadata = MetaData()

# 2. Create the Table to the Database
census = Table('census', metadata,
               Column('state', String(30)),
               Column('sex', String(1)),
               Column('age', Integer()),
               Column('pop2000', Integer()),
               Column('pop2008', Integer()))
metadata.create_all(engine)

# 3. Read the Data from the CSV
values_list = []
for row in csv_reader:
    data = {'state': row[0], 'sex': row[1], 'age':row[2], 'pop2000': row[3], 'pop2008': row[4]}
    values_list.append(data)

# 4. Load Data from a list into the Table
stmt = insert(census)
results = connection.execute(stmt, values_list)

# 5. Example Queries
# 5.1 Determine the Average Age by Population
# (first determine the average age weighted by the population in 2008, and then group by sex)
stmt = select([census.columns.sex,
               (func.sum(census.columns.pop2008 * census.columns.age) /
                func.sum(census.columns.pop2008)).label('average_age')
               ])
stmt = stmt.group_by(census.columns.sex)
results = connection.execute(stmt).fetchall()
for result in results:
    print(result.sex, result.average_age)

# 5.2 Determine the Percentage of Population by Gender and State
# (determine the percentage of the population in 2000 that comprised of women, grouping by state)
stmt = select([census.columns.state,
    (func.sum(
        case([
            (census.columns.sex == 'F', census.columns.pop2000)
        ], else_=0)) /
     cast(func.sum(census.columns.pop2000), Float) * 100).label('percent_female')
])
stmt = stmt.group_by(census.columns.state)
results = connection.execute(stmt).fetchall()
for result in results:
    print(result.state, result.percent_female)

# 5.3 Determine the Difference by State from the 2000 and 2008 Censuses
# (calculate the states that changed the most in population, limiting the query to display only the top 10 states)
stmt = select([census.columns.state,
     (census.columns.pop2008-census.columns.pop2000).label('pop_change')
])
stmt = stmt.group_by(census.columns.state)
stmt = stmt.order_by(desc('pop_change'))
stmt = stmt.limit(10)
results = connection.execute(stmt).fetchall()
for result in results:
    print('{}:{}'.format(result.state, result.pop_change))