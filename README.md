# Tholkapian_N_PATP_exp03
# Experiment 3: Logistic Regression for Classification

## Aim 
To develop a Logistic Regression classification model using Python and machine learning libraries to predict categorical outcomes based on input features and evaluate the model's performance using accuracy, confusion matrix, and classification report.


## Algorithm
1. Start.
2. Import the required Python libraries.
3. Load the dataset and separate the independent variables (X) and dependent variable (y).
4. Split the dataset into training and testing sets using an 80:20 ratio.
5. Standardize the input features using StandardScaler.
6. Train the Logistic Regression model using the training dataset.
7. Predict the target values using the test dataset.
8. Evaluate the model using accuracy score, confusion matrix, and classification report.
9. Display the evaluation results.
10. Stop.


## Input
```
# Step 1: Import required libraries
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load the dataset using scikit-learn's loader directly to access .data and .feature_names
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display classification report
report = classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)
```
## Output
```

Accuracy: 0.9736842105263158

Confusion Matrix:
[[41  2]
 [ 1 70]]

Classification Report:
              precision    recall  f1-score   support

   malignant       0.98      0.95      0.96        43
      benign       0.97      0.99      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
```

## Result
Thus, Linear Discriminant Analysis was successfully implemented on the Iris dataset. The classification accuracy was calculated, and the three flower species were visualized using two linear discriminant components (LD1 and LD2).
