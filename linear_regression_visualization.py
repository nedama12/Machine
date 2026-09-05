import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from LinearRegressionGrades import df, model


def create_regression_graph():
    """Generate a graph from the real dataset and trained model."""
    data = df.sort_values("Sunlight_Hours")
    predictions = model.predict(data[["Sunlight_Hours"]])

    plt.figure(figsize=(10, 5))
    plt.scatter(
        df["Sunlight_Hours"],
        df["Solar_Energy"],
        s=12,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.3,
        label="Real Data",
    )
    plt.plot(
        data["Sunlight_Hours"],
        predictions,
        color="red",
        label="Regression Line",
    )
    plt.title("Solar Energy Production vs. Sunlight Hours")
    plt.xlabel("Sunlight Hours")
    plt.ylabel("Solar Energy (kWh/day)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/regression_graph.png")
    plt.close()
