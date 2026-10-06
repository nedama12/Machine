import math
from pathlib import Path

from flask import Flask, render_template, request

from LinearRegressionGrades import df, model, predict_energy
from logistic_regression import df as credit_df, predict_credit
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
from kmeans_clustering import (
    df as social_df,
    df_results as social_results_df,
    feature_columns,
    predict_cluster,
)
from kmeans_visualization import create_kmeans_graph
from kmeans_metrics import (
    MANUAL_EXERCISE,
    get_cluster_distribution,
    get_cluster_profiles,
    get_silhouette_interpretation,
    get_silhouette_score,
)
from qlearning_gridworld import (
    ACTIONS,
    BATCH_SIZE,
    EPISODES,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    GAMMA,
    GRID_COLS,
    GRID_ROWS,
    LEARNING_RATE,
    MAX_STEPS_EVALUATION,
    MAX_STEPS_TRAINING,
    REWARD_TABLE,
    SEED,
    build_grid,
    cell_counts,
    run_session,
)

app = Flask(__name__)

# Build the static visualization from the current K-Means results.
create_kmeans_graph()


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
        income=income,
        record_count=len(credit_df)
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

@app.route("/unsupervised/concepts")
def unsupervised_concepts():
    return render_template("unsupervised_concepts.html")


@app.route("/unsupervised/manual")
def unsupervised_manual():
    image_names = [
        "kmeans_initial.png",
        "kmeans_iteration_1.png",
        "kmeans_iteration_2.png",
        "kmeans_iteration_3.png",
    ]
    available_images = [
        image for image in image_names
        if (Path(app.static_folder) / image).is_file()
    ]
    return render_template(
        "kmeans_manual.html",
        exercise=MANUAL_EXERCISE,
        images=available_images,
    )


@app.route("/unsupervised/application", methods=["GET", "POST"])
def unsupervised_application():
    error = None
    prediction = None
    input_columns = [
        ("age", "Age"),
        ("social_hours", "Daily_Social_Media_Hours"),
        ("screen_hours", "Daily_Screen_Time_Hours"),
        ("posts", "Posts_Per_Week"),
        ("messages", "Messages_Per_Day"),
    ]
    form_values = {
        name: "" for name, _ in input_columns
    }
    bounds = {}
    for name, column in input_columns:
        bounds[name] = (
            float(social_df[column].min()),
            float(social_df[column].max()),
        )
    range_description = (
        f"Age {bounds['age'][0]:g} to {bounds['age'][1]:g}, "
        f"social media hours {bounds['social_hours'][0]:g} to {bounds['social_hours'][1]:g}, "
        f"screen time {bounds['screen_hours'][0]:g} to {bounds['screen_hours'][1]:g}, "
        f"posts per week {bounds['posts'][0]:g} to {bounds['posts'][1]:g}, "
        f"and messages per day {bounds['messages'][0]:g} to {bounds['messages'][1]:g}."
    )
    page_size = 50
    try:
        records_page = max(1, int(request.args.get("page", "1")))
    except ValueError:
        records_page = 1
    page_count = math.ceil(len(social_results_df) / page_size)
    records_page = min(records_page, page_count)
    start = (records_page - 1) * page_size
    record_frame = social_results_df.iloc[start:start + page_size]
    clustered_records = [
        {
            "record": int(index) + 1,
            "age": row["Age"],
            "social_hours": row["Daily_Social_Media_Hours"],
            "screen_hours": row["Daily_Screen_Time_Hours"],
            "posts": row["Posts_Per_Week"],
            "messages": row["Messages_Per_Day"],
            "cluster": int(row["Cluster"]),
        }
        for index, row in record_frame.iterrows()
    ]

    if request.method == "POST":
        form_values = {key: request.form.get(key, "").strip() for key in form_values}
        try:
            values = {}
            for key, _ in input_columns:
                value = float(form_values[key])
                if not math.isfinite(value):
                    raise ValueError
                minimum, maximum = bounds[key]
                if not minimum <= value <= maximum:
                    raise ValueError
                values[key] = value

            if any(values[key] != int(values[key]) for key in ("age", "posts", "messages")):
                raise ValueError

            prediction = predict_cluster(
                values["age"],
                values["social_hours"],
                values["screen_hours"],
                values["posts"],
                values["messages"],
            )
        except (TypeError, ValueError):
            error = (
                "Enter valid values within the dataset ranges: "
                f"{range_description} Age, posts, and messages must be whole numbers."
            )

    return render_template(
        "kmeans_application.html",
        error=error,
        prediction=prediction,
        form_values=form_values,
        record_count=len(social_df),
        feature_count=len(feature_columns),
        profiles=get_cluster_profiles(),
        bounds=bounds,
        silhouette=get_silhouette_score(),
        clustered_records=clustered_records,
        records_page=records_page,
        page_count=page_count,
        records_total=len(social_results_df),
    )


@app.route("/unsupervised/metrics")
def unsupervised_metrics():
    return render_template(
        "kmeans_metrics.html", record_count=len(social_df),
        silhouette=get_silhouette_score(), cluster_summary=get_cluster_distribution(),
        profiles=get_cluster_profiles(),
        silhouette_interpretation=get_silhouette_interpretation(),
    )


@app.route("/reinforcement-learning/concepts")
def reinforcement_learning_concepts():
    return render_template(
        "reinforcement_learning_concepts.html",
        rows=GRID_ROWS,
        cols=GRID_COLS,
        rewards=REWARD_TABLE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )


@app.route("/reinforcement-learning/application", methods=["GET", "POST"])
def reinforcement_learning_application():
    session = None
    path = None

    if request.method == "POST":
        session = run_session()
        path = session["evaluation"]["path"]

    return render_template(
        "reinforcement_learning_application.html",
        grid=build_grid(path),
        counts=cell_counts(),
        rows=GRID_ROWS,
        cols=GRID_COLS,
        actions=ACTIONS,
        rewards=REWARD_TABLE,
        config={
            "episodes": EPISODES,
            "gamma": GAMMA,
            "epsilon_start": EPSILON_START,
            "epsilon_min": EPSILON_MIN,
            "epsilon_decay": EPSILON_DECAY,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_steps_training": MAX_STEPS_TRAINING,
            "max_steps_evaluation": MAX_STEPS_EVALUATION,
            "seed": SEED,
        },
        session=session,
    )


if __name__ == "__main__":
    app.run(debug=True)
