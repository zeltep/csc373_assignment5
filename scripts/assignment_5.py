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

    name = "SGD"
    estimator = SGDRegressor(random_state=seed)

    pipeline = Pipeline(steps=[
        ('col_transformer', col_transformer),
        ('estimator', estimator)
    ])

    pipeline.fit(train_X, train_y)
    
    preds = pipeline.predict(dev_X)

    scores = {}
    scores['r2'] = pipeline.score(train_X, train_y)
    scores['mse'] = mean_squared_error(dev_y, preds)
    scores['over'], scores['under'] = utils.over_under_pred_scores(dev_y, preds)

    output_handle.write(f"{name}:\n")
    utils.write_estimation_scores(output_handle, scores)


def classification(train_X, dev_X, train_y, dev_y, output_handle, model=None):
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
    

    if model is not None:
        model.fit(train_X, train_y)
        scores = {}
        scores['accuracy'] = accuracy_score(dev_y, preds)
        scores['over'], scores['under'] = utils.over_under_pred_scores(dev_y, preds, classification=True)
    
        utils.write_classification_scores(output_handle, scores)
        
        return


    best_acc = 0
    best_model = None

    for name, classifier in classifiers.items():
        
        pipeline = Pipeline(steps=[
            ('col_transformer', col_transformer),
            ('classifier', classifier)
        ])

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

    joblib.dump(best_model, "best_model.pkl")
    return best_model
    

def main():
    #dataset = utils.get_dataset()
    #df = utils.create_df(dataset)
    
    df = utils.get_subsample()

    #utils.report_df(df)
    
    cleaned_df = utils.clean_data(df)
    X, y = utils.create_target(cleaned_df)
    
    #Original Estimation
    train_X, dev_X, train_y, dev_y = train_test_split(X, y, random_state=seed)
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
    binary_y = utils.binarize_y(y)    
    train_X, dev_X, train_y, dev_y = train_test_split(X, binary_y, random_state=seed)
    handle = open("../output/model_scores/classification/original.txt", "w+")
    best_clsf_model = classification(train_X, dev_X, train_y, dev_y, handle)
    handle.close()

      
    #
    #utils.subsample_df(df)



if __name__ == "__main__":
    main()
