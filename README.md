# Churn_modeling_ANN

# 🧠 Customer Churn Prediction using Artificial Neural Network (ANN)

This project implements an **Artificial Neural Network (ANN)** using **TensorFlow/Keras** to predict customer churn from the **Churn_Modelling** dataset.  
The goal is to determine whether a customer is likely to leave the service based on various demographic and account features.

---

## 📁 Project Structure

Churn_Modelling_ANN/
│
├── Churn_Modelling_ANN_model.ipynb # Main Jupyter Notebook
├── Churn_Modelling.csv # Dataset used
├── README.md # Project documentation
└── requirements.txt # List of dependencies

---

## 🚀 Project Overview

Customer churn prediction is a critical problem in the banking and telecom industries.  
By training an ANN, this project aims to **classify customers as churned (1)** or **retained (0)** based on attributes like geography, gender, age, balance, and activity level.

---

## ⚙️ Steps Involved

### 1️⃣ Data Preprocessing
- Handled missing values using **SimpleImputer**
- Encoded categorical variables (*Geography*, *Gender*) using **OneHotEncoder**
- Applied **feature scaling** with **StandardScaler**

### 2️⃣ Model Building
- Built a **Sequential Neural Network** using **TensorFlow/Keras**
- Used:
  - **ReLU** activation for hidden layers  
  - **Sigmoid** activation for output layer  
- Compiled with:
  - **Optimizer:** Adam  
  - **Loss Function:** Binary Crossentropy  
  - **Metrics:** Accuracy  

### 3️⃣ Model Training
```python
ann.fit(X_train, y_train, batch_size=32, epochs=100)
-batch_size=32: Number of samples processed before each weight update
-epochs=100: Number of full passes through the training dataset

4️⃣ Evaluation
-Evaluated model on the test set
-Compared predicted vs. actual values
-Analyzed confusion matrix and accuracy score

📊 Results
-The ANN achieved high accuracy on the test dataset, showing good generalization capability.
-It effectively learned patterns that distinguish churned and non-churned customers.

🧩 Technologies Used
Python 3.10+
TensorFlow / Keras
NumPy
Pandas
scikit-learn
Matplotlib / Seaborn (optional for visualization)

📂 Dataset
The dataset used: Churn_Modelling.csv
It contains the following features:
Customer ID, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, and Exited.
