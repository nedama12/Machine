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


# ---- Use cases ----
# Use Case 1 and Use Case 2 are still pending; the "Use Cases" nav dropdown
# already lists them as "Coming soon". Once their templates are ready, add
# routes here following the same pattern as use_case_3 / use_case_4 and
# enable their links in templates/base.html.
#
# @app.route("/use-cases/1")
# def use_case_1():
#     return render_template("use_case_1.html")
#
# @app.route("/use-cases/2")
# def use_case_2():
#     return render_template("use_case_2.html")


@app.route("/use-cases/3")
def use_case_3():
    return render_template("use_case_3.html")


@app.route("/use-cases/4")
def use_case_4():
    return render_template("use_case_4.html")


if __name__ == "__main__":
    app.run(debug=True)
