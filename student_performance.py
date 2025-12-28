import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import argparse
import sys


def load_data(path="data.csv"):
    """Load CSV data into a DataFrame."""
    return pd.read_csv(path)


def train_model(df):
    """Train a Linear Regression model and return it along with the test MSE."""
    X = df[["study_hours", "attendance"]]
    y = df["score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse


def predict_score(model, study_hours, attendance):
    """Return the predicted score given a model and inputs."""
    input_data = pd.DataFrame([[study_hours, attendance]], columns=["study_hours", "attendance"])
    pred = model.predict(input_data)
    return float(pred[0])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train model and optionally predict a score.")
    parser.add_argument("--study-hours", type=float, help="Study hours for prediction")
    parser.add_argument("--attendance", type=float, help="Attendance percentage for prediction")
    parser.add_argument("--data", type=str, default="data.csv", help="Path to CSV data file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    df = load_data(args.data)
    model, mse = train_model(df)
    print("Mean Squared Error:", mse)

    # If both CLI args provided, use them. Otherwise, try interactive input if available.
    if args.study_hours is not None and args.attendance is not None:
        pred = predict_score(model, args.study_hours, args.attendance)
        print("Predicted Score:", round(pred, 2))
        return

    # If running in a TTY, prompt the user. Otherwise, avoid blocking and print usage.
    if sys.stdin.isatty():
        try:
            study_hours = float(input("Enter study hours: "))
            attendance = float(input("Enter attendance percentage: "))
            pred = predict_score(model, study_hours, attendance)
            print("Predicted Score:", round(pred, 2))
        except EOFError:
            print("No input available. Provide --study-hours and --attendance for non-interactive runs.")
    else:
        print("No interactive input available. To predict, re-run with --study-hours and --attendance.")


if __name__ == "__main__":
    main()

