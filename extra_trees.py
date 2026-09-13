import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

df = pd.read_csv("credit_risk.csv")

X = df[
    [
        "Monthly_Income",
        "Monthly_Debt",
        "Credit_Score",
        "Age"
    ]
]

y = df["Credit_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = ExtraTreesClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


def predict_risk(
    monthly_income,
    monthly_debt,
    credit_score,
    age
):

    prediction = model.predict(
        pd.DataFrame({
            "Monthly_Income": [monthly_income],
            "Monthly_Debt": [monthly_debt],
            "Credit_Score": [credit_score],
            "Age": [age]
        })
    )

    return prediction[0]


def process_risk_prediction(form):

    monthly_income = form.get("monthly_income", "").strip()
    monthly_debt = form.get("monthly_debt", "").strip()
    credit_score = form.get("credit_score", "").strip()
    age = form.get("age", "").strip()

    try:

        income_value = float(monthly_income)
        debt_value = float(monthly_debt)
        score_value = float(credit_score)
        age_value = int(age)

        if not 1300000 <= income_value <= 6000000:
            return None, "Monthly income must be between 1,300,000 and 6,000,000 COP.", monthly_income, monthly_debt, credit_score, age

        if not 200000 <= debt_value <= 4800000:
            return None, "Monthly debt must be between 200,000 and 4,800,000 COP.", monthly_income, monthly_debt, credit_score, age

        if not 300 <= score_value <= 850:
            return None, "Credit score must be between 300 and 850.", monthly_income, monthly_debt, credit_score, age

        if not 18 <= age_value <= 70:
            return None, "Age must be between 18 and 70.", monthly_income, monthly_debt, credit_score, age

        prediction = int(
            predict_risk(
                income_value,
                debt_value,
                score_value,
                age_value
            )
        )

        return (
            prediction,
            None,
            monthly_income,
            monthly_debt,
            credit_score,
            age
        )

    except ValueError:

        return (
            None,
            "Please enter valid numeric values.",
            monthly_income,
            monthly_debt,
            credit_score,
            age
        )