import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

prediction_years_count = 20

# Dataset loading and visualization
dataset = pd.read_csv("master.csv")
print(dataset.head())
print(dataset.columns)

# Load relevant series and subsets
dataset_br = dataset[dataset.country == "Brazil"]
years = dataset_br.year.unique()
annual_events = [dataset_br[dataset_br.year == y].suicides_no.sum() for y in years]

# Prediction
reg = linear_model.LinearRegression()
reg.fit(years.reshape(-1, 1), annual_events)
prediction_years = np.arange(max(years), max(years) + prediction_years_count)
prediction = reg.predict(prediction_years.reshape(-1, 1))

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(years, annual_events)
plt.plot(prediction_years, prediction)
plt.ylabel("Events")
plt.xlabel("Year")
plt.title("Suicides in Brazil by Year - General")
plt.legend(["Official", "Estimated"])
plt.show()
