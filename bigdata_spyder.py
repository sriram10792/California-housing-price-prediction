# -*- coding: utf-8 -*-
"""
Created on Sun May  6 18:23:03 2018

@author: Ramesh Narayanan
"""


import  findspark
findspark.init()
import pyspark as ps
import warnings
from pyspark.sql import SQLContext


from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("project")
sc = SparkContext(conf = conf)


# Import SparkSession
from pyspark.sql import SparkSession

# Build the SparkSession
spark = SparkSession.builder \
   .master("local") \
   .appName("Linear Regression Model") \
   .getOrCreate()
   
sc = spark.sparkContext


# Load in the data
rdd = sc.textFile('C:/Users/Ramesh Narayanan/Downloads/cal_housing/CaliforniaHousing/cal_housing.data')

# Load in the header
header = sc.textFile('C:/Users/Ramesh Narayanan/Downloads/cal_housing/CaliforniaHousing/cal_housing.domain')


header.collect()


# Split lines on commas
rdd = rdd.map(lambda line: line.split(","))

# Inspect the first 2 lines 
rdd.take(2)


# Import the necessary modules 
from pyspark.sql import Row

# Map the RDD to a DF
df = rdd.map(lambda line: Row(longitude=line[0], 
                              latitude=line[1], 
                              housingMedianAge=line[2],
                              totalRooms=line[3],
                              totalBedRooms=line[4],
                              population=line[5], 
                              households=line[6],
                              medianIncome=line[7],
                              medianHouseValue=line[8])).toDF()


# Import all from `sql.types`
from pyspark.sql.types import *

# Write a custom function to convert the data type of DataFrame columns
def convertColumn(df, names, newType):
  for name in names: 
     df = df.withColumn(name, df[name].cast(newType))
  return df 

# Assign all column names to `columns`
columns = ['households', 'housingMedianAge', 'latitude', 'longitude', 'medianHouseValue', 'medianIncome', 'population', 'totalBedRooms', 'totalRooms']

# Conver the `df` columns to `FloatType()`
df = convertColumn(df, columns, FloatType())


# Import all from `sql.functions` 
from pyspark.sql.functions import *

# Adjust the values of `medianHouseValue`
df = df.withColumn("medianHouseValue", col("medianHouseValue")/100000)

# Show the first 2 lines of `df`
df.take(2)


# Import all from `sql.functions` if you haven't yet
from pyspark.sql.functions import *

# Divide `totalRooms` by `households`
roomsPerHousehold = df.select(col("totalRooms")/col("households"))

# Divide `population` by `households`
populationPerHousehold = df.select(col("population")/col("households"))

# Divide `totalBedRooms` by `totalRooms`
bedroomsPerRoom = df.select(col("totalBedRooms")/col("totalRooms"))

# Add the new columns to `df`
df = df.withColumn("roomsPerHousehold", col("totalRooms")/col("households")) \
   .withColumn("populationPerHousehold", col("population")/col("households")) \
   .withColumn("bedroomsPerRoom", col("totalBedRooms")/col("totalRooms"))
   
# Inspect the result
df.first()



# Re-order and select columns
df = df.select("medianHouseValue", 
              "totalBedRooms", 
              "population",
              "housingMedianAge",
              "households", 
              "medianIncome", 
              "roomsPerHousehold", 
            "populationPerHousehold", 
              "bedroomsPerRoom")





import pyspark.mllib
import pyspark.mllib.regression
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import *





# Import `DenseVector`
from pyspark.ml.linalg import DenseVector

# Define the `input_data` 
input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

# Replace `df` with the new DataFrame
dfnew = spark.createDataFrame(input_data, ["label", "features"])




# Import `StandardScaler` 
from pyspark.ml.feature import StandardScaler

# Initialize the `standardScaler`
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

# Fit the DataFrame to the scaler
scaler = standardScaler.fit(dfnew)

# Transform the data in `df` with the scaler
scaled_df = scaler.transform(dfnew)

# Inspect the result
scaled_df.take(2)



# Split the data into train and test sets
train_data, test_data = scaled_df.randomSplit([.8,.2],seed=1234)



# Import `LinearRegression`
from pyspark.ml.regression import LinearRegression

# Initialize `lr`
lr = LinearRegression(labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the data to the model
linearModel = lr.fit(train_data)


# Generate predictions
predicted = linearModel.transform(test_data)

# Extract the predictions and the "known" correct labels
predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
labels = predicted.select("label").rdd.map(lambda x: x[0])

# Zip `predictions` and `labels` into a list
predictionAndLabel = predictions.zip(labels).collect()

# Print out first 5 instances of `predictionAndLabel` 
predictionAndLabel[:5]


# Coefficients for the model
linearModel.coefficients

# Intercept for the model
linearModel.intercept


# Get the RMSE
linearModel.summary.rootMeanSquaredError

# Get the R2
linearModel.summary.r2



from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator


# Re-order and select columns
df4 = df.select("medianHouseValue",
               "totalBedRooms", 
              "population",
              "housingMedianAge",
              "households", 
              "medianIncome", 
              "roomsPerHousehold", 
            "populationPerHousehold", 
              "bedroomsPerRoom",)


# Import `DenseVector`
from pyspark.ml.linalg import DenseVector

# Define the `input_data` 
input_data = df3.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

# Replace `df` with the new DataFrame
df3 = spark.createDataFrame(input_data, ["label", "features"])



df4.toPandas().head()

features=df4.rdd.map(lambda row: row[1:])

features.take(1)

from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.ml.linalg import Vectors
from pyspark.mllib.feature import StandardScaler


standardizer = StandardScaler()
model = standardizer.fit(features)
features_transform = model.transform(features)


labels=df4.rdd.map(lambda row: row[0])
labels.take(2)


transformedData=labels.zip(features_transform)
transformedData.take(2)



transformedData = transformedData.map(lambda row : LabeledPoint(row[0],row[1]))
transformedData.take(5)
type(transformedData)


trainingData, testingData = transformedData.randomSplit([.8,.2])

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils


model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                        numTrees=3, featureSubsetStrategy="auto",
                                        impurity='variance', maxDepth=4, maxBins=32)


predictions = model.predict(testingData.map(lambda x: x.features))


labelsAndPredictions = testingData.map(lambda lp: lp.label).zip(predictions)

trainingMSE = labelsAndPredictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() /\
    float(trainingData.count())
    
testMSE = labelsAndPredictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() /\
    float(testingData.count())
    
    
print('Train Mean Squared Error = ' + str(trainingMSE))
print('Test Mean Squared Error = ' + str(testMSE))


from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils


model = GradientBoostedTrees.trainRegressor(trainingData,
                                            categoricalFeaturesInfo={}, numIterations=3)

predictions = model.predict(testingData.map(lambda x: x.features))
labelsAndPredictions = testingData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() /\
    float(testingData.count())
    
    
    
    
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression GBT model:')
print(model.toDebugString())


model = GradientBoostedTrees.trainRegressor(trainingData,
                                            categoricalFeaturesInfo={}, numIterations=100)


predictions = model.predict(testingData.map(lambda x: x.features))
labelsAndPredictions = testingData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() /\
    float(testingData.count())
    
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression GBT model:')