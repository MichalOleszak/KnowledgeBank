# Intro to Spark -------------------------------------------------------------------------------------------------------

# Spark is a platform for cluster computing. Spark lets you spread data and computations over clusters with multiple
# nodes (think of each node as a separate computer). Splitting up your data makes it easier to work with very large
# datasets because each node only works with a small amount of data.

# The first step in using Spark is connecting to a cluster. In practice, the cluster will be hosted on a remote machine
# that's connected to all other nodes. There will be one computer, called the master that manages splitting up the data
# and the computations. The master is connected to the rest of the computers in the cluster, which are called slaves.
# The master sends the slaves data and calculations to run, and they send their results back to the master.

# Creating the connection is as simple as creating an instance of the SparkContext class. The class constructor takes
# a few optional arguments that allow you to specify the attributes of the cluster you're connecting to.
import pyspark
sc = SparkContext()
# Verify SparkContext
print(sc)
# Print Spark version
print(sc.version)

# Spark's core data structure is the Resilient Distributed Dataset (RDD). This is a low level object that lets Spark
# work its magic by splitting data across multiple nodes in the cluster. However, RDDs are hard to work with directly,
# so in this course you'll be using the Spark DataFrame abstraction built on top of RDDs.
# The Spark DataFrame was designed to behave a lot like a SQL table (a table with variables in the columns and
# observations in the rows). Not only are they easier to understand, DataFrames are also more optimized for complicated
# operations than RDDs.
# When you start modifying and combining columns and rows of data, there are many ways to arrive at the same result,
# but some often take much longer than others. When using RDDs, it's up to the data scientist to figure out the right
# way to optimize the query, but the DataFrame implementation has much of this optimization built in!
# To start working with Spark DataFrames, you first have to create a SparkSession object from your SparkContext.
# You can think of the SparkContext as your connection to the cluster and the SparkSession as your interface with that
# connection.

# Creating multiple SparkSessions and SparkContexts can cause issues, so it's best practice to use the
# SparkSession.builder.getOrCreate() method. This returns an existing SparkSession if there's already one in the
# environment, or creates a new one if necessary!

# Creatinga Spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Viewing tables
print(spark.catalog.listTables())

# Running SQL queries on the cluster
query = "FROM flights SELECT * LIMIT 10"
flights10 = spark.sql(query)
flights10.show()

# Pandafy a Spark DataFrame
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"
flight_counts = spark.sql(query)
pd_counts = flight_counts.toPandas()

# Putting pandas DataFrames into a Spark cluster
# The .createDataFrame() method takes a pandas DataFrame and returns a Spark DataFrame.
# The output of this method is stored locally, not in the SparkSession catalog. This means that you can use all the
# Spark DataFrame methods on it, but you can't access the data in other contexts.
# For example, a SQL query (using the .sql() method) that references your DataFrame will throw an error. To access the
# data in this way, you have to save it as a temporary table.
# You can do this using the .createTempView() Spark DataFrame method, which takes as its only argument the name of the
# temporary table you'd like to register. This method registers the DataFrame as a table in the catalog, but as this
# table is temporary, it can only be accessed from the specific SparkSession used to create the Spark DataFrame.
# There is also the method .createOrReplaceTempView(). This safely creates a new temporary table if nothing was there
# before, or updates an existing table if one was already defined.
# Create df with  random numbers
pd_temp = pd.DataFrame(np.random.random(10))
# Make it a Spark DataFrame
spark_temp = spark.createDataFrame(pd_temp)
# Add spark_temp to the catalog under the name "temp"
spark_temp.createOrReplaceTempView("temp")

# Reading in data to Spark without using pandas
file_path = "/usr/local/share/datasets/airports.csv"
airports = spark.read.csv(file_path, header=True)


# Manipulating data with pyspark.sql -----------------------------------------------------------------------------------
# Creating columns
flights = spark.table("flights")
flights = flights.withColumn("duration_hrs", flights.duration_mins / 60)

# Filtering data
# Filter flights with a SQL string
long_flights1 = flights.filter("distance > 1000")
# Filter flights with a boolean column
long_flights2 = flights.filter(flights.distance > 1000)

# Selecting
# Using column names as strings
selected1 = flights.select("tailnum", "origin", "dest")
# Using df.col notation
temp = flights.select(flights.origin, flights.dest, flights.carrier)
# The difference between .select() and .withColumn() methods is that .select() returns only the columns you specify,
# while .withColumn() returns all the columns of the DataFrame in addition to the one you defined.
# Similar to SQL, you can also use the .select() method to perform column-wise operations. When you're selecting
# a column using the df.colName notation, you can perform any column operation and the .select() method will return
# the transformed column.
flights.select(flights.air_time/60)
# You can also use the .alias() method to rename a column you're selecting.
flights.select((flights.air_time/60).alias("duration_hrs"))
# The equivalent Spark DataFrame method .selectExpr() takes SQL expressions as a string.
flights.selectExpr("air_time/60 as duration_hrs")
# with the SQL as keyword being equivalent to the .alias() method. To select multiple columns, pass multiple strings.
# Define avg_speed
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")
# Select the correct columns
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)
# Create the same table using a SQL expression
speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed")

# Aggregating
# Find the minimum value of a column, col, in a DataFrame, df
df.groupBy().min("col").show()
# Find the shortest flight from PDX in terms of distance
flights.filter(flights.origin == "PDX").groupBy().min("distance").show()
# Find the longest flight from SEA in terms of duration
flights.filter(flights.origin == "SEA").groupBy().max("air_time").show()
# Average duration of Delta flights
flights.filter(flights.carrier == "DL").filter(flights.origin == "SEA").groupBy().avg("air_time").show()
# Total hours in the air of all planes in the data set
flights.withColumn("duration_hrs", flights.air_time/60).groupBy().sum("duration_hrs").show()

# Grouping and aggregating
# Part of what makes aggregating so powerful is the addition of groups. PySpark has a whole class devoted to grouped
# data frames: pyspark.sql.GroupedData. They can be created by calling the .groupBy() method on a DataFrame with no
# arguments. When you pass the name of one or more columns in your DataFrame to the .groupBy() method, the aggregation
# methods behave like when you use a GROUP BY statement in a SQL query!
# Number of flights each plane made
by_plane = flights.groupBy("tailnum")
by_plane.count().show()
# Average duration of flights by origin airport
by_origin = flights.groupBy("origin")
by_origin.avg("air_time").show()
# In addition to the GroupedData methods, there is also the .agg() method. This method lets you pass an aggregate
# column expression that uses any of the aggregate functions from the pyspark.sql.functions submodule. This submodule
# contains many useful functions for computing things like standard deviations. All the aggregation functions in this
# submodule take the name of a column in a GroupedData table.
import pyspark.sql.functions as F
# Standard deviation of the departure delay by month and destination
by_month_dest.avg("dep_delay").show()
by_month_dest.agg(F.stddev("dep_delay")).show()

# Joining
# Rename column faa in airports to dest, as it is called in flights
airports = airports.withColumnRenamed("faa", "dest")
flights_with_airports = flights.join(airports, on="dest", how="leftouter")


# Machine Learning Pipelines -------------------------------------------------------------------------------------------
