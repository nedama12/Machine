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


if __name__ == "__main__":
    app.run(debug=True)
