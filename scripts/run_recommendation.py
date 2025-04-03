import os, sys
import utils
from pyspark.ml import Pipeline

def main():
    args = sys.argv[1:]

    if len(args) != 2:
        print("Please enter model file path and data file path")
        return
    elif not os.path.exists(args[0]):
        print("Invalid model file path")
        return
    elif not os.path.exists(args[1])
        print("Invalid data file path")
        return

    dataset = utils.get_dataset_from_path(args[1])
    df = utils.create_df(dataset)
    df = utils.clean_data(df)

    df['log_hours'] = utils.get_log_y(df['hours'])

    spark = SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(df)

    model = Pipeline.load(args[0])
    predictions = model.predict(spark_df)
    predictions = predictions.toPandas()

    scores = {}
    trues = predictions['log_hours']
    preds = predictions['prediction']
    scores['mse'] = mean_squared_error(trues, preds)
    scores['over'], scores['under'] = utils.over_under_pred_scores(trues, preds)
     
    print(f"MSE: {scores['mse']:.3f")
    print(f"Overprediction: {scores['over']:.3f")
    print(f"Underprediction: {scores['under']:.3f")

    

if __name__ == "__main__":
    main()
