from flask import Flask, render_template

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
