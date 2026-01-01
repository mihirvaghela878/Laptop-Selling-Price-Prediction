import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("laptop_data.csv")

# Features and target
X = df[
    [
        "SSD",
        "ram",
        "resolution_width",
        "resolution_height",
        "cpu_name",
        "inches",
        "cpu_speed",
        "weight_kg",
        "hdd",
        "ispanel",
        "touchscreen"
    ]
]

y = df["price"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cpu", OneHotEncoder(handle_unknown="ignore"), ["cpu_name"])
    ],
    remainder="passthrough"
)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
preds = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
pickle.dump(pipeline, open("model.pkl", "wb"))
print("Model saved as model.pkl")
