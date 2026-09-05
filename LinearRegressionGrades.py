import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("solar_energy_colombia.csv", sep=";", decimal=",")

X = df[["Sunlight_Hours"]]
y = df["Solar_Energy"]

model = LinearRegression()
model.fit(X, y)

def predict_energy(hours):
    prediction = model.predict(
        pd.DataFrame({"Sunlight_Hours": [hours]})
    )
    return prediction[0]
