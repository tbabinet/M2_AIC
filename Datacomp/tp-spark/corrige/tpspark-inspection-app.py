from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.functions import desc
from pyspark.sql.types import *

spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
spark.sparkContext.setLogLevel("WARN") 

df1 = spark.read.load("/home/tp-home002/bgroz/testspark/dataset2/food_inspections.csv",format="csv",sep=",", inferSchema="true", header="true")

inspections = df1.na.drop()

def labelForResults(s):
    if s == 'Fail':
        return 0.0
    elif s == 'Pass w/ Conditions' or s == 'Pass':
        return 1.0
    else:
        return -1.0


monudf = udf(labelForResults,DoubleType())
labeledData = inspections.select(monudf(inspections["results"]).alias('label'),"violations").where('label >= 0')


splitdf = labeledData.randomSplit([0.25, 0.75],105)
training = splitdf[0]
validationDf = splitdf[1]

tokenizer = Tokenizer(inputCol="violations", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.01)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])


model = pipeline.fit(labeledData)


predictionsDf = model.transform(validationDf)
predictionsDf.registerTempTable('Predictions')
predictionsDf.columns


numSuccesses = predictionsDf.where("""(prediction = 0 AND results = 'Fail') OR
                                        (prediction = 1 AND (results = 'Pass' OR
                                                            results = 'Pass w/ Conditions'))""").count()
numInspections = predictionsDf.count()


with open('result-tpinspection.txt','r+') as resfile:
     resfile.write("There were"+ str(numInspections)+ "inspections and there were"+ str(numSuccesses)+ "successful predictions")
     resfile.write("This is a"+str((float(numSuccesses) / float(numInspections)) * 100) + "%"+ "success rate")