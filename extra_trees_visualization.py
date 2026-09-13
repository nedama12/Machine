import matplotlib.pyplot as plt

from extra_trees import df


def create_extra_trees_graph():

    low_risk = df[df["Credit_Risk"] == 0]
    high_risk = df[df["Credit_Risk"] == 1]

    plt.scatter(
        low_risk["Credit_Score"],
        low_risk["Monthly_Debt"],
        s=20,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.3,
        label="Low Risk"
    )

    plt.scatter(
        high_risk["Credit_Score"],
        high_risk["Monthly_Debt"],
        s=20,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.3,
        label="High Risk"
    )

    plt.title("Credit Risk Classification")
    plt.xlabel("Credit Score")
    plt.ylabel("Monthly Debt (COP)")
    plt.legend()

    plt.savefig("static/credit_risk_graph.png")
    plt.close()