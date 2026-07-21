import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("../data/Advertising.csv")

# Remove first column if it is unnamed
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])

# Features
X = data[['TV', 'Radio', 'Newspaper']]

# Target
y = data['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
print("R2 Score:", r2_score(y_test, prediction))

print("MAE:", mean_absolute_error(y_test, prediction))

# Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, prediction)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.savefig("../images/prediction.png")

plt.show()