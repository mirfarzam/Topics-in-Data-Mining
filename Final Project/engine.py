import csv
import os
import sys
from csv import reader
# Spark RDD and SQL imports
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc
# Spark MLlib package imports
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
##
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import PCA, VectorAssembler, StandardScaler, StringIndexer
from pyspark.mllib.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.mllib.feature import Normalizer
import timeit
## logestic regression for classification
from pyspark.ml.classification import LogisticRegression
##
import numpy as np 
##
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.util import MLUtils

#Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def count_df(filename):
    '''
    Write a Python script using DataFrames that prints the number of trees
    (non-header lines) in the data file passed as first argument.
    Test file: tests/test_count_df.py
    Note: The return value should be an integer
    '''
    spark = init_spark()
    init_df = spark.read.option("inferSchema", "true").option("header", "true").csv(filename,header=True)
    cols = init_df.columns
    cols = cols[:-1]


    vecAssembler = VectorAssembler(inputCols=cols, outputCol="features")
    standardizer = StandardScaler(withMean=True, withStd=True,inputCol='features',outputCol='std_features')
    indexer = StringIndexer(inputCol="class", outputCol="label_idx")
    pca = PCA(k=5, inputCol="std_features", outputCol="pca")

    lr_pca = LogisticRegression(featuresCol='pca', labelCol='label_idx')

    lr_withoutpp = LogisticRegression(featuresCol='pca', labelCol='label_idx')

    pipeline = Pipeline(stages=[vecAssembler, standardizer, indexer, pca, lr_withoutpp])

    train, test = init_df.randomSplit([0.7, 0.3])

    model = pipeline.fit(train)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prediction = model.transform(test)
    
    score = prediction.select(['prediction', 'label_idx'])

    metrics = BinaryClassificationMetrics(score.rdd)
    print(metrics)

    score.show(n=score.count())

    acc = score.rdd.map(lambda x: x[1] == x[0]).sum() / score.count()
    print(acc)

    # print(acc)


    # pipe = Pipeline(stages=[vecAssembler, pca])
    # model = pipe.fit(init_df)
    # result = model.transform(init_df)
    # features = result.select("pcaFeatures").rdd.map(lambda x: np.array(x))
    # labels = result.select("class").collect()
    # # ADD YOUR CODE HERE
    # # raise Exception("Not implemented yet")
    # # Create initial LogisticRegression model
    # lr = LogisticRegression(labelCol=, featuresCol="class", maxIter=10)



count_df("./pima-indians-diabetes.csv")
