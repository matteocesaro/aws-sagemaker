
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import sklearn
import joblib
import argparse
import joblib
import os
import pandas as pd

# Tis function is necessary for loading your model 
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir,"model.joblib"))
    return clf

if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    #hyperparameters sent by the client are passed as command-line arguments
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    #Data, model and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")

    args, _ = parser.parse_known_args()

    print("SKLearn Version", sklearn.__version__)
    print("joblib Version", joblib.__version__)

    print("[INFO] Reading data")
    print()
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop(-1)

    print("Building training and testing dataset")
    print()
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    print("Column order: ")
    print(features)
    print()

    print("Label column is: ", label)
    print()

    print("Data Shape: ")
    print()
    print("--- SHAPE OF TRAINING DATA (85%) --- ")
    print(X_train.shape)
    print(y_train.shape)
    print()
    print("--- SHAPE OF TESTING DATA (15%) --- ")
    print(X_test.shape)
    print(y_test.shape)
    print()

    print("Training RandomForest Model")
    print()
    model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_state)
    model.fit(X_train,y_train)
    print()

    model_path = os.path.join(args.model_dir, "model.joblib") 
    joblib.dump(model, model_path)
    print("Model persisted at " + model_path)
    print()

    y_pred = model.predict(X_test)
    test_err = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print()
    print("--- METRIC RESULTS FOR TESTING DATA ---")
    print()
    print("Total Rows are: ", X_test.shape[0])
    print(f"MAE:  {test_err:,.0f} EUR")
    print(f"R²:   {test_r2:.4f}")
