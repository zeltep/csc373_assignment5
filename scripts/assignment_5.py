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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib



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


def classification(df):
    X, y = utils.create_target(df)
    # Binarizes y for above and below median, 1 if above, 0 if below
    y = (y > y.median()).astype(int)
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

    classifiers = {
        "Dummy": DummyClassifier(),
        "Logistic Regression": LogisticRegression(solver = 'saga'),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    best_acc = 0
    best_model = None
    for name, classifier in classifiers.items():
        
        pipeline = Pipeline(steps=[
            ('col_transformer', col_transformer),
            ('classifier', classifier)
        ])

        pipeline.fit(train_X, train_y)
        preds = pipeline.predict(dev_X)
        accuracy = accuracy_score(dev_y, preds)

        over_pred_score = sum((preds == 1) & (dev_y == 0)) / len(dev_y)
        under_pred_score = sum((preds == 0) & (dev_y == 1)) / len(dev_y)        

        print(f"\nClassifier: {name}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Overprediction Rate (dev): {over_pred_score:.2f}")
        print(f"Underprediction Rate (dev): {under_pred_score:.2f}\n")    
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = pipeline

    joblib.dump(best_model, "best_model.pkl")
    

def main():
    dataset = utils.get_dataset()
    df = utils.create_df(dataset)
    
    #df = utils.get_subsample()
    utils.report_df(df)
    
    cleaned_df = utils.clean_data(df)
    
    estimation(cleaned_df)
    classification(cleaned_df)
    #utils.subsample_df(df)

    '''sample_df = utils.subsample_df(df)
    cleaned_sample = utils.clean_data(sample_df)
    estimation(cleaned_sample)
    classification(cleaned_sample)'''



if __name__ == "__main__":
    main()
