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

def estimation(df):
    X, y = utils.create_target(df)
    train_X, dev_X, train_y, dev_y = train_test_split(X, y)
    
    num_cols, cat_cols = utils.sort_num_cat_cols(train_X)
    
    col_transformer = ColumnTransformer(transformers=([
        ('cat', OneHotEncoder(), cat_cols), 
        ('num', StandardScaler(), num_cols)
    ]))

    estimator = SGDRegressor()

    pipeline = Pipeline(steps=[
        ('transformer', col_transformer),
        ('estimator', estimator)
    ])

    pipeline.fit(train_X, train_y)
    preds = pipeline.predict(dev_X)
    
    print(f"R-Squared: {pipeline.score(train_X, train_y)}")
    print(f"MSE (dev): {mean_squared_error(dev_y, preds)}")
    
    

def main():
    #dataset = utils.get_dataset()
    #df = utils.create_df(dataset)
    df = utils.get_subsample()
    utils.report_df(df)
    
    cleaned_df = utils.clean_data(df)
    
    estimation(cleaned_df)
    
    #utils.subsample_df(df)


if __name__ == "__main__":
    main()
