'''
----------------------
Evan Zelt & Brett Westerberg
CSC 373: Assignment 5
Dr. Khuri
Wake Forest University
3/26/2025
----------------------

Main file for model training/selection/saving
'''
import sys, os
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline as MLPipeline
from pyspark.ml.recommendation import ALS 
from pyspark.ml.feature import StringIndexer
import joblib

seed=1

def estimation(train_X, dev_X, train_y, dev_y, output_handle):
    num_cols, text_cols, cat_cols = utils.sort_cols(train_X)
            
    vectorizer = TfidfVectorizer(stop_words='english', min_df=0.1)
    text_preprocessor = Pipeline(steps=[
        ('vectorizer', vectorizer),
    ])

    col_transformer = ColumnTransformer(transformers=([
        ('cat', OneHotEncoder(), cat_cols), 
        ('text', text_preprocessor, "text"),
        ('num', StandardScaler(), num_cols)
    ]))

    estimators = {
        "SGD": SGDRegressor(random_state=seed)
    }

    pipeline = Pipeline(steps=[
        ('col_transformer', col_transformer),
        ('estimator', None)
    ])

    best_model = None
    best_mse = None
    for name, estimator in estimators.items():
        pipeline.set_params(estimator=estimator)

        pipeline.fit(train_X, train_y)
        
        preds = pipeline.predict(dev_X)

        scores = {}
        scores['r2'] = pipeline.score(train_X, train_y)
        scores['mse'] = mean_squared_error(dev_y, preds)
        scores['over'], scores['under'] = utils.over_under_pred_scores(dev_y, preds)

        output_handle.write(f"{name}:\n")
        utils.write_estimation_scores(output_handle, scores)
        
        if best_mse is None or scores['mse'] < best_mse:
            best_mse = scores['mse']
            best_model = pipeline

    joblib.dump(best_model, "../output/models/estimation_pipeline.pkl")
    return best_model


def classification(train_X, dev_X, train_y, dev_y, output_handle, model=None):
    if model is not None:
        model.fit(train_X, train_y)
        preds = model.predict(dev_X)

        scores = {}
        scores['accuracy'] = accuracy_score(dev_y, preds)
        scores['over'], scores['under'] = utils.over_under_pred_scores(dev_y, preds, classification=True)
    
        output_handle.write("Scores from best model:\n")
        utils.write_classification_scores(output_handle, scores)
        
        return

    num_cols, text_cols, cat_cols = utils.sort_cols(train_X)

    vectorizer = TfidfVectorizer(stop_words='english', min_df=0.1)
    text_preprocessor = Pipeline(steps=[
        ('vectorizer', vectorizer),
    ])

    col_transformer = ColumnTransformer(transformers=([
        ('cat', OneHotEncoder(), cat_cols), 
        ('text', text_preprocessor, "text"),
        ('num', StandardScaler(), num_cols)
    ]))

    classifiers = {
        "Dummy": DummyClassifier(),
        "Logistic Regression": LogisticRegression(solver = 'saga', random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed)
    }

    best_acc = 0
    best_model = None

    pipeline = Pipeline(steps=[
        ('col_transformer', col_transformer),
        ('classifier', None)
    ])

    for name, classifier in classifiers.items():
        pipeline.set_params(classifier=classifier)    

        pipeline.fit(train_X, train_y)
        preds = pipeline.predict(dev_X)

        scores = {}
        scores['accuracy'] = accuracy_score(dev_y, preds)
        scores['over'], scores['under'] = utils.over_under_pred_scores(dev_y, preds, classification=True)
    
        output_handle.write(f"{name}:\n")
        utils.write_classification_scores(output_handle, scores)

        if scores['accuracy'] > best_acc:
            best_acc = scores['accuracy']
            best_model = pipeline

    joblib.dump(best_model, "../output/models/classification_pipeline.pkl")
    return best_model
    

def recommendation(df, output_handle):
    df['log_hours'] = utils.get_log_y(df['hours'])

    spark = SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(df)

    indexer = StringIndexer(inputCol="username", outputCol="user_code", handleInvalid='keep')
    recommendation = ALS(maxIter=5, regParam=0.01, userCol='user_code', itemCol='product_id', ratingCol='log_hours', nonnegative=True, seed=seed)

    pipeline = MLPipeline(stages=[indexer, recommendation])

    model = pipeline.fit(spark_df)
    
    predictions = model.transform(spark_df) 
    predictions = predictions.toPandas()
    
    scores = {}
    trues = predictions['log_hours']
    preds = predictions['prediction']
    scores['mse'] = mean_squared_error(trues, preds)
    scores['over'], scores['under'] = utils.over_under_pred_scores(trues, preds)
    
    output_handle.write("Recommendation scores:\n")
    utils.write_recommendation_scores(output_handle, scores)

    joblib.dump(model, "../output/models/recommendation_pipeline.pkl")


def main():
    dataset = utils.get_dataset()
    df = utils.create_df(dataset)
    
    #df = utils.get_subsample()

    utils.report_df(df)
    
    cleaned_df = utils.clean_data(df)
    X, y = utils.create_target(cleaned_df)

    #Original Estimation
    train_X, dev_X, train_y, dev_y = train_test_split(X, y, random_state=seed, test_size=0.2)
    handle = open("../output/model_scores/estimation/original.txt", "w+")
    estimation(train_X, dev_X, train_y, dev_y, handle)
    handle.close()

    #Removed outliers
    ro_train_X, ro_train_y = utils.remove_outliers(train_X, train_y)
    handle = open("../output/model_scores/estimation/removed_outliers.txt", "w+")
    estimation(ro_train_X, dev_X, ro_train_y, dev_y, handle)
    handle.close()

    #Log2 Scaled
    train_log_y = utils.get_log_y(train_y) 
    dev_log_y = utils.get_log_y(dev_y) 
    handle = open("../output/model_scores/estimation/log2.txt", "w+")
    estimation(train_X, dev_X, train_log_y, dev_log_y, handle)
    handle.close()

    #Original Classification
    train_bin_y = utils.binarize_y(train_y)
    dev_bin_y = utils.binarize_y(dev_y)
    handle = open("../output/model_scores/classification/original.txt", "w+")
    best_clsf_model = classification(train_X, dev_X, train_bin_y, dev_bin_y, handle)
    handle.close()

    #Classification: split by years
    early_X, early_y = utils.get_data_before(X, y, 2014)
    late_X, late_y = utils.get_data_after(X, y, 2015)

    early_bin_y = utils.binarize_y(early_y)
    late_bin_y = utils.binarize_y(late_y)

    #Proactive Classification (training on data in 2014 or before) 
    handle = open("../output/model_scores/classification/proactive.txt", "w+")
    classification(early_X, late_X, early_bin_y, late_bin_y, handle, model=best_clsf_model)
    handle.close()

    #Retroactive Classification (training on data in 2015 or later)
    handle = open("../output/model_scores/classification/retroactive.txt", "w+")
    classification(late_X, early_X, late_bin_y, early_bin_y, handle, model=best_clsf_model)
    handle.close()

    #Recommendation 
    log_df = cleaned_df.copy()
    handle = open("../output/model_scores/recommendation/recommendation.txt", "w+")
    recommendation(log_df, handle) 
    handle.close()


if __name__ == "__main__":
    main()
