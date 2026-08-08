# 🌸 Iris Flower Classification

A beginner-friendly Machine Learning project that predicts the species of an Iris flower based on its **sepal and petal measurements**.

The project uses the **Iris dataset** and a Machine Learning classification algorithm to classify flowers into three species:

* 🌱 Iris Setosa
* 🌱 Iris Versicolor
* 🌱 Iris Virginica

---

## 🎯 Objective

The main objective of this project is to build a Machine Learning model that can predict the species of an Iris flower using:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

This project also helped me understand the basic Machine Learning workflow, including data loading, data analysis, visualization, model training, evaluation, prediction, and model saving.

---

## 📊 Dataset

The project uses the **Iris Flower Dataset**, which contains **150 samples** and **5 columns**.

### Features

| Feature      | Description           |
| ------------ | --------------------- |
| Sepal Length | Length of the sepal   |
| Sepal Width  | Width of the sepal    |
| Petal Length | Length of the petal   |
| Petal Width  | Width of the petal    |
| Species      | Target flower species |

The dataset contains three classes:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

Each class contains 50 samples.

---

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Joblib**
* **Jupyter Notebook / VS Code**
* **Git & GitHub**

---

## 🤖 Machine Learning

The project follows a basic Machine Learning pipeline:

```text
Dataset
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Data Visualization
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
New Flower Prediction
   ↓
Save Model
```

---

## 🔍 Exploratory Data Analysis

The dataset was analyzed to understand the relationships between different flower measurements.

The project includes:

* Dataset shape
* Dataset information
* Statistical summary
* Missing value checking
* Class distribution
* Correlation analysis
* Pair plot
* Heatmap
* Count plot

### Dataset Shape

```text
(150, 5)
```

### Missing Values

```text
No missing values found
```

---

## 📈 Data Visualization

The following visualizations were created:

### 1. Count Plot

Used to understand the distribution of the three Iris species.

### 2. Pair Plot

Used to visualize relationships between different features and identify how the three species are separated.

### 3. Correlation Heatmap

Used to understand the correlation between numerical features.

---

## 🧠 Model Training

The dataset is divided into training and testing data.

```python
X = df.drop("species", axis=1)
y = df["species"]
```

The data is then split using `train_test_split()`.

The Machine Learning model is trained using the training data and evaluated using the test data.

---

## 🔮 Prediction

After training the model, it can predict the species of a new flower based on its measurements.

Example:

```python
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print(prediction)
```

Example output:

```text
['setosa']
```

---

## 💾 Saving the Model

The trained model can be saved using **Joblib**.

```python
import joblib

joblib.dump(model, "iris_model.pkl")
```

The saved model can later be loaded without training the model again.

```python
model = joblib.load("iris_model.pkl")
```

---

## 📁 Project Structure

```text
Iris-Flower-Classification/
│
├── iris_classification.py
├── iris_model.pkl
├── iris.csv
├── requirements.txt
├── README.md
│
└── images/
    ├── countplot.png
    ├── pairplot.png
    └── heatmap.png
```

If you are using a Jupyter Notebook, you can instead have:

```text
Iris-Flower-Classification/
│
├── Iris_Flower_Classification.ipynb
├── iris_model.pkl
├── requirements.txt
├── README.md
│
└── images/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Iris-Flower-Classification.git
```

Move into the project directory:

```bash
cd Iris-Flower-Classification
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the Python program:

```bash
python iris_classification.py
```

Or open the Jupyter Notebook:

```text
Iris_Flower_Classification.ipynb
```

and run the cells sequentially.

---

## 📦 Requirements

Example `requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

## 📊 Results

The trained Machine Learning model successfully classifies Iris flowers into their respective species.

The model performance can be evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report

Example:

```text
Accuracy: XX%
```

> Replace `XX%` with the actual accuracy obtained from your model.

---

## 🚀 Future Improvements

Possible improvements for this project include:

* Creating a Streamlit web application
* Adding multiple Machine Learning algorithms
* Comparing model performance
* Adding user input for flower measurements
* Deploying the application online
* Creating a graphical user interface

---


