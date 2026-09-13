from flask import Flask, render_template, request

from LinearRegressionGrades import df, model, predict_energy
from logistic_regression import predict_credit
from extra_trees import process_risk_prediction, df as risk_df
from logistic_regression_metrics import (
    matrix,
    accuracy,
    precision,
    recall,
    f1
)
from extra_trees_metrics import (
    matrix as risk_matrix,
    accuracy as risk_accuracy,
    precision as risk_precision,
    recall as risk_recall,
    f1 as risk_f1
)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/definition")
def definition():
    return render_template("definition.html")


@app.route("/types")
def types():
    return render_template("types.html")


@app.route("/linear-regression/concepts")
def linear_regression_concepts():
    return render_template(
        "linear_regression_concepts.html",
        coefficient=float(model.coef_[0]),
        intercept=float(model.intercept_),
    )


@app.route("/linear-regression/application", methods=["GET", "POST"])
def linear_regression_application():

    prediction = None
    error = None
    hours = ""

    if request.method == "POST":

        hours = request.form.get("hours", "").strip()

        if not hours:
            error = "Please enter the sunlight hours."

        else:
            try:
                hours_value = float(hours)

                if not 4 <= hours_value <= 8:
                    error = "Sunlight hours must be between 4 and 8 hours."

                else:
                    prediction = float(
                        predict_energy(hours_value)
                    )

            except ValueError:
                error = "Please enter a valid numeric value."

    return render_template(
        "linear_regression_application.html",
        prediction=prediction,
        error=error,
        hours=hours,
        record_count=len(df),
        coefficient=float(model.coef_[0]),
        intercept=float(model.intercept_),
        example_prediction=float(predict_energy(7)),
    )
@app.route("/logistic-regression/concepts")
def logistic_regression_concepts():
    return render_template("logistic_regression_concepts.html")

@app.route(
    "/logistic-regression/application",
    methods=["GET", "POST"]
)
def logistic_regression_application():

    prediction = None
    error = None
    income = ""

    if request.method == "POST":

        income = request.form.get("income", "").strip()

        if not income:
            error = "Please enter the monthly income."

        else:
            try:
                income_value = float(income)

                if not 1300000 <= income_value <= 6000000:
                    error = (
                        "Monthly income must be between "
                        "1,300,000 and 6,000,000 COP."
                    )

                else:
                    prediction = int(
                        predict_credit(income_value)
                    )

            except ValueError:
                error = "Please enter a valid numeric value."

    return render_template(
        "logistic_regression_application.html",
        prediction=prediction,
        error=error,
        income=income
    )

@app.route("/logistic-regression/metrics")
def logistic_regression_metrics():
    return render_template(
        "logistic_regression_metrics.html",
        matrix=matrix,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1
    )


@app.route("/use-cases/1")
def use_case_1():
    return render_template("use_case_1.html")


@app.route("/use-cases/2")
def use_case_2():
    return render_template("use_case_2.html")


@app.route("/use-cases/3")
def use_case_3():
    return render_template("use_case_3.html")


@app.route("/use-cases/4")
def use_case_4():
    return render_template("use_case_4.html")

@app.route("/extra-trees/concepts")
def extra_trees_concepts():
    return render_template("extra_trees_concepts.html")

@app.route(
    "/extra-trees/application",
    methods=["GET", "POST"]
)
def extra_trees_application():

    prediction = None
    error = None

    monthly_income = ""
    monthly_debt = ""
    credit_score = ""
    age = ""

    if request.method == "POST":

        (
            prediction,
            error,
            monthly_income,
            monthly_debt,
            credit_score,
            age
        ) = process_risk_prediction(request.form)

    return render_template(
        "extra_trees_application.html",
        prediction=prediction,
        error=error,
        monthly_income=monthly_income,
        monthly_debt=monthly_debt,
        credit_score=credit_score,
        age=age,
        record_count=len(risk_df)
    )
@app.route("/extra-trees/metrics")
def extra_trees_metrics():
    return render_template(
        "extra_trees_metrics.html",
        matrix=risk_matrix,
        accuracy=risk_accuracy,
        precision=risk_precision,
        recall=risk_recall,
        f1=risk_f1
    )

if __name__ == "__main__":
    app.run(debug=True)