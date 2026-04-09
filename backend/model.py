from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier()

# Dummy training data
X = [
    [50000, 20000, 50, 0.6, 1000],
    [20000, 18000, 20, 0.1, 900],
    [70000, 30000, 60, 0.5, 1200],
    [15000, 14000, 10, 0.05, 800]
]

y = [1, 0, 1, 0]

# Train model
model.fit(X, y)

# Prediction function
def predict_user(features):
    return model.predict([features])[0]