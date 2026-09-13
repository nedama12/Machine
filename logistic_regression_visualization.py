import matplotlib.pyplot as plt

from logistic_regression import df


def create_logistic_graph():
    plt.scatter(
        df["Monthly_Income"],
        df["Credit_Approval"],
        s=20,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.3
    )

    plt.title("Credit Approval vs. Monthly Income")
    plt.xlabel("Monthly Income (COP)")
    plt.ylabel("Credit Approval")
    plt.yticks([0, 1], ["Rejected", "Approved"])

    plt.savefig("static/credit_approval_graph.png")
    plt.close()