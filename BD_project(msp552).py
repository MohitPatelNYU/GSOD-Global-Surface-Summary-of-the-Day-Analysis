from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('big_data_project').getOrCreate()
print('Program Started')
df = spark.read.csv('US_Weather_Min_2.csv', header=True, inferSchema=True)
print('data loaded')
df = df.withColumnRenamed('rain_drizzle', 'label')

num_cols = ['temp','dewp','max','min','prcp','fog','hail','snow_ice_pellets','thunder']
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

print('stages begin')
stages = []
assemblerInputs=num_cols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


from pyspark.ml import Pipeline
print('pipeline begin')
cols = df.columns
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['features']+cols
df = df.select(selectedCols)
train, test = df.randomSplit([0.67, 0.33])
print('pipeline done')

print('Count of Training Data',train.count())
print('Count of Test Data',test.count())

from pyspark.ml.classification import LogisticRegression
print('Linear Regression Started')
LR = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=15)
LR_model = LR.fit(train)
print('Linear Regression Completed')

trainingSummary = LR_model.summary
roc = trainingSummary.roc.toPandas()
print('Training set ROC for LR: ' + str(trainingSummary.areaUnderROC))

from pyspark.ml.evaluation import BinaryClassificationEvaluator
predictions_LR = LR_model.transform(test)
evaluator = BinaryClassificationEvaluator()
print("Area Under ROC:for Binary Classification " + str(evaluator.evaluate(predictions_LR, {evaluator.metricName: "areaUnderROC"})))

from pyspark.ml.classification import DecisionTreeClassifier
print('Decision Tree Started')
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)
predictions.select('temp', 'slp', 'label', 'rawPrediction', 'prediction').show(100)

print('Decision Tree Completed')

evaluator = BinaryClassificationEvaluator()

print("Test Area Under ROC: for Decision Tree " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
