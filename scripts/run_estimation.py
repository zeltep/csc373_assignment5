import os, sys
import utils
import joblib

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
    cleaned_df = utils.clean_data(df)
    X, y = utils.create_target(cleaned_df)

    model = joblib.load(args[0])
    preds = model.predict(X)
     
    scores = {}
    scores['mse'] = mean_squared_error(y, preds)
    scores['over'], scores['under'] = utils.over_under_pred_scores(y, preds)
    
    print(f"MSE: {scores['mse']:.3f")
    print(f"Overprediction: {scores['over']:.3f")
    print(f"Underprediction: {scores['under']:.3f")

    

if __name__ == "__main__":
    main()
