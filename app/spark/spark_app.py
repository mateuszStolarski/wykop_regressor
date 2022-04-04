
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
import pyspark.sql.functions as F
from pyspark.ml.functions import array_to_vector
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from consts import *


def get_session() -> SparkSession:
    # conf = SparkConf().setAppName(APP_NAME).setMaster(MASTER)
    # context = SparkContext(conf=conf)
    # context.setLogLevel(LOG_LEVEL)

    session = SparkSession.builder \
        .appName(APP_NAME) \
        .master(MASTER)\
        .config(INPUT_STRING, CONNECTION_STRING) \
        .config(OUTPUT_STRING, CONNECTION_STRING) \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.0")\
        .getOrCreate()

    return session


def grid_search(train):
    base_model = LinearRegression(maxIter=15)
    params = ParamGridBuilder()\
        .addGrid(base_model.regParam, [0.1, 0.2, 0.3, 0.4, 0.5]) \
        .addGrid(base_model.fitIntercept, [False, True])\
        .addGrid(base_model.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0])\
        .build()

    tvs = TrainValidationSplit(estimator=base_model,
                               estimatorParamMaps=params,
                               evaluator=RegressionEvaluator(),
                               trainRatio=0.8)

    grid_result = tvs.fit(train)
    best_model = grid_result.bestModel
    print("Best model hyperparameters:")
    print('regParam: ', best_model._java_obj.getRegParam())
    print('fitItercept: ', best_model._java_obj.getFitIntercept())
    print('elasticNetParam: ', best_model._java_obj.getElasticNetParam())
    print("")

    return best_model


def evaluate_model(model, test, printing_label):
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
    preds = model.transform(test).select("label", "prediction")

    r2_score = evaluator.evaluate(preds, {evaluator.metricName: "mae"})
    rmse = evaluator.evaluate(preds, {evaluator.metricName: "rmse"})

    print(f"{printing_label} RMSE: {rmse:.3f}")
    print(f"{printing_label} MAE: {r2_score:.3f}")


def main():
    spark = get_session()
    df = spark.read.format('mongo').load()
    df.printSchema()


main()
