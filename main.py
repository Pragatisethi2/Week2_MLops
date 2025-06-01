import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os

# Load dataset
data = pd.read_csv('iris.csv')

# Split into train and test
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train['species']
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test['species']

# Train model
mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
mod_dt.fit(X_train, y_train)

# Make predictions
prediction = mod_dt.predict(X_test)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(mod_dt, "model/model.joblib")

# Evaluate and save metrics
accuracy = metrics.accuracy_score(y_test, prediction)
conf_matrix = metrics.confusion_matrix(y_test, prediction)

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'metric': ['accuracy'],
    'value': [accuracy]
})
metrics_df.to_csv("metrics.csv", index=False)

# Optional: Save confusion matrix
conf_matrix_df = pd.DataFrame(conf_matrix,
                              index=mod_dt.classes_,
                              columns=mod_dt.classes_)
conf_matrix_df.to_csv("confusion_matrix.csv")
