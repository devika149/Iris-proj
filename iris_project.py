import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# Load dataset
iris = load_iris()

print("Dataset loaded successfully!")

print("\nDataset Keys:")
print(iris.keys())

# Convert dataset into DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add species column
df["Species"] = iris.target

# Display first five rows
print("\nFirst 5 Records:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSpecies Count:")

sns.countplot(x="Species", data=df)

plt.title("Number of Flowers in Each Species")

plt.show()
sns.pairplot(df, hue="Species")

plt.show()
plt.figure(figsize=(8,6))

sns.heatmap(df.corr(), annot=True)

plt.title("Correlation Heatmap")

plt.show()
# Input features
X = df.iloc[:, :-1]

# Output (Target)
y = df["Species"]

print("\nInput Features (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())


# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)
import joblib

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Save the trained model
joblib.dump(knn, "iris_model.pkl")

print("\nKNN Model Trained Successfully!")
print("Model saved successfully!")
# Predict on test data
y_pred = knn.predict(X_test)

print("\nPredicted Species:")
print(y_pred)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Create Decision Tree model
dt = DecisionTreeClassifier(random_state=42)

# Train the model
dt.fit(X_train, y_train)

# Make predictions
dt_pred = dt.predict(X_test)

# Calculate accuracy
dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:", dt_accuracy)
# Create SVM model
svm = SVC()

# Train the model
svm.fit(X_train, y_train)

# Make predictions
svm_pred = svm.predict(X_test)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_pred)

print("\nSVM Accuracy:", svm_accuracy)
print("\n========== Model Comparison ==========")

print("KNN Accuracy           :", accuracy)
print("Decision Tree Accuracy :", dt_accuracy)
print("SVM Accuracy           :", svm_accuracy)
print("\nEnter the flower measurements:")
iris = load_iris()

print("===== Iris Flower Prediction =====")

# Get input from user
sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))

# Create DataFrame with correct column names
new_flower = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=iris.feature_names
)

# Predict the species
prediction = knn.predict(new_flower)

# Display result
print("\nPredicted Species:", iris.target_names[prediction[0]])


