import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("credit_approval.csv")

X = df[["Monthly_Income"]]
y = df["Credit_Approval"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

model.fit(X_train, y_train)


def predict_credit(income):
    prediction = model.predict(
        pd.DataFrame({"Monthly_Income": [income]})
    )

    return prediction[0]