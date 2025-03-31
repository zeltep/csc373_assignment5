'''
----------------------
Evan Zelt
CSC 373: Assignment 5
Dr. Khuri
Wake Forest University
3/26/2025
----------------------

Main file for model training/selection/saving
'''
import sys, os
import utils
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def estimation(df):
    X, y = utils.create_target(df)
    train_X, dev_X, train_y, dev_y = train_test_split(X, y)
    
    num_cols, text_cols, cat_cols = utils.sort_cols(train_X)
            
    text_preprocessor = Pipeline(steps=[
        ('vectorizer', TfidfVectorizer()),
        #('decomp', TruncatedSVD())
    ])

    col_transformer = ColumnTransformer(transformers=([
        ('cat', OneHotEncoder(), cat_cols), 
        ('text', text_preprocessor, "text"),
        ('num', StandardScaler(), num_cols)
    ]))

    
    estimator = SGDRegressor()

    pipeline = Pipeline(steps=[
        ('col_transformer', col_transformer),
        ('estimator', estimator)
    ])

    pipeline.fit(train_X, train_y)
    
    preds = pipeline.predict(dev_X)
    
    over_pred_score, under_pred_score = utils.over_under_pred_scores(dev_y, preds)
    
    print(f"R-Squared: {pipeline.score(train_X, train_y):.3f}")
    print(f"MSE (dev): {mean_squared_error(dev_y, preds):.3f}")
    print(f"Overprediction Rate (dev): {over_pred_score:.2f}")
    print(f"Underprediction Rate (dev): {under_pred_score:.2f}")
    
    
    

def main():
    dataset = utils.get_dataset()
    df = utils.create_df(dataset)
    #df = utils.get_subsample()
    utils.report_df(df)
    
    cleaned_df = utils.clean_data(df)
    
    estimation(cleaned_df)
    
    #utils.subsample_df(df)


if __name__ == "__main__":
    main()
