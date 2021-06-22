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

# Reading csv to dataframe while setting data types
schema = StructType([
  StructField("col1", StringType(), nullable=False),
  StructField("col2", ByteType(), nullable=True)
])
df = spark.read.options(header="true").schema(schema).csv("file.csv")

# Skip invalid rows
df = spark.read.options(header="true", mode="DROPMALFORMED").csv("file.csv")


# Manipulating data with pyspark.sql -----------------------------------------------------------------------------------
# Creating columns
flights = spark.table("flights")
flights = flights.withColumn("duration_hrs", flights.duration_mins / 60)

# Conditionally replace values
df = df.withColumn("column", when(col("column") > 5, None).otherwise(col("column")))

# Filtering data
# Filter flights with a SQL string
long_flights1 = flights.filter("distance > 1000")
# Filter flights with a boolean column
long_flights2 = flights.filter(flights.distance > 1000)
long_flights1 = flights.filter(col("distance") > 1000)

# Selecting
# Using column names as strings
selected1 = flights.select("tailnum", "origin", "dest")
# Using df.col notation
temp = flights.select(flights.origin, flights.dest, flights.carrier)
# Rename
df.select(col("old").alias("new"))
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
from pyspark.ml import Pipeline

# At the core of the pyspark.ml module are the Transformer and Estimator classes. Almost every other class in the module 
# behaves similarly to these two basic classes.
# 	* 	Transformer classes have a .transform() method that takes a DataFrame and returns a new DataFrame; usually 
#		the original one with a new column appended. For example, you might use the class Bucketizer to create discrete 
#		bins from a continuous feature or the class PCA to reduce the dimensionality of your dataset using principal 
#		component analysis.
#	*	Estimator classes all implement a .fit() method. These methods also take a DataFrame, but instead of returning 
#		another DataFrame they return a model object. This can be something like a StringIndexerModel for including 
#		categorical data saved as strings in your models, or a RandomForestModel that uses the random forest algorithm 
#		for classification or regression.

# Spark only handles numeric data. That means all of the columns in your DataFrame must be either integers or decimals 
# (called 'doubles' in Spark). When we imported our data, we let Spark guess what kind of information each column held. 
# Unfortunately, Spark doesn't always guess right and you can see that some of the columns in our DataFrame are strings 
# containing numbers as opposed to actual numeric values. To remedy this, you can use the .cast() method in combination 
# with the .withColumn() method. It's important to note that .cast() works on columns, while .withColumn() works on 
# DataFrames. The only argument you need to pass to .cast() is the kind of value you want to create, in string form. 
# For example, to create integers, you'll pass the argument "integer" and for decimal numbers you'll use "double".
# You can put this call to .cast() inside a call to .withColumn() to overwrite the already existing column

# Cast the columns to integers
# dataframe = dataframe.withColumn("col", dataframe.col.cast("new_type"))
model_data = model_data.withColumn("air_time", model_data.air_time.cast('integer'))

# Create new column
model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)

# Create a boolean
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)
model_data = model_data.withColumn("label", model_data.is_late.cast('integer'))

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")

# One-hot encoding of categorical feaures
# The first step to encoding your categorical feature is to create a StringIndexer. Members of this class are Estimators 
# that take a DataFrame with a column of strings and map each unique string to a number. Then, the Estimator returns 
# a Transformer that takes a DataFrame, attaches the mapping to it as metadata, and returns a new DataFrame with a numeric 
# column corresponding to the string column.
#
# The second step is to encode this numeric column as a one-hot vector using a OneHotEncoder. This works exactly the same 
# way as the StringIndexer by creating an Estimator and then a Transformer. The end result is a column that encodes your 
# categorical feature as a vector that's suitable for machine learning routines!
#
# The last step in the Pipeline is to combine all of the columns containing our features into a single column. 
# This has to be done before modeling can take place because every Spark modeling routine expects the data to be in this 
# form. You can do this by storing each of the values from a column as an entry in a vector. Then, from the model's point 
# of view, every observation is a vector that contains all of the information about it and a label that tells the modeler 
# what value that observation corresponds to. Because of this, the pyspark.ml.feature submodule contains a class called 
# VectorAssembler. This Transformer takes all of the columns you specify and combines them into a new vector column.

# Create a StringIndexer
carr_indexer = StringIndexer(inputCol='carrier', outputCol='carrier_index')
# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol='carrier_index', outputCol='carrier_fact')
# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "plane_age"], outputCol='features')
# Make the pipeline
flights_pipe = Pipeline(stages=[carr_indexer, carr_encoder, vec_assembler])
# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)

# Train test split
# In Spark it's important to make sure you split the data after all the transformations. This is because operations 
# like StringIndexer don't always produce the same index even when given the same list of strings.
training, test = piped_data.randomSplit([.6, .4])


# Model Tuning & Selection ---------------------------------------------------------------------------------------------
from pyspark.ml.classification import LogisticRegression
import pyspark.ml.evaluation as 
import pyspark.ml.tuning as tune

# Create a LogisticRegression Estimator
lr = LogisticRegression()
# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")
# Create the parameter grid
grid = tune.ParamGridBuilder()
# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])
# Build the grid
grid = grid.build()
# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
               			 estimatorParamMaps=grid,
               			 evaluator=evaluator)
# Fit cross validation models
models = cv.fit(training)
# Extract the best model
best_lr = models.bestModel
# Use the model to predict the test set
test_results = best_lr.transform(test)
# Evaluate the predictions
print(evaluator.evaluate(test_results))


# Deploying spark applications ----------------------------------------------------------------------------------------
spark-submit --master "local[*]" --py-files a.py,b.py MAIN_PYTHON_FILE app_arguments
# --master [optional] - URL to cluster manager or "local[*]" - where to get resources from
# --py-files - comma-separated list of zi[, egg or py files to be copied to the worker nodes
# app_arguments -- args taken by MAIN_PYTHON_FILE

# Prepare zip file to pass to --py0files
zip --recurse-paths outputfilename.zip folder-to-compress
spark-submit --py-files outputfilename.zip folder-to-compress/script.py


# Unit tests for PySpark ----------------------------------------------------------------------------------------------
# Providing input for data transformation test cases -- constructing data frames in-memory
from pyspark.sql import Row

purchase = Row("price", "qunatity", "product")
record = purchase(12.23, 1, "abc")
df = spark.createDataFrame((record,))
