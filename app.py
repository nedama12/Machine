from pathlib import Path
import os

os.chdir(Path(__file__).resolve().parent)

from flask import Flask, render_template, request

from LinearRegressionGrades import df, model, predict_energy
from linear_regression_visualization import create_regression_graph


app = Flask(__name__)

example_prediction = float(predict_energy(7))

create_regression_graph()


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

                error = (
                    "Please enter a valid numeric value "
                    "for sunlight hours."
                )

    return render_template(
        "linear_regression_application.html",
        prediction=prediction,
        error=error,
        hours=hours,
        record_count=len(df),
        coefficient=float(model.coef_[0]),
        intercept=float(model.intercept_),
        example_prediction=example_prediction,
    )