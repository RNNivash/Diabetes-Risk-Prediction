# 🩺 Diabetes Risk Prediction

This project focuses on predicting the risk of diabetes using a machine learning pipeline — from data preprocessing and exploratory analysis to building multiple classification models and comparing their performance.

<img width="1280" alt="Image" src="https://github.com/user-attachments/assets/ed442e57-10fe-4bb5-ae96-4afa97d84af1" />---

## 📂 Dataset

The dataset used is `diabetes_prediction_dataset.csv`, containing features like:

- **age**
- **gender**
- **bmi** (Body Mass Index)
- **HbA1c_level** (Glycated hemoglobin)
- **blood_glucose_level**
- **smoking_history**
- and **diabetes** (target variable)

---

## ⚙️ Project Workflow

### 1. Data Cleaning & Preprocessing
- Dropped rare entries like `'Other'` in gender.
- Encoded categorical variables (`gender`, `smoking_history`) using `LabelEncoder`.
- Detected and capped outliers for `bmi`, `blood_glucose_level`, and `HbA1c_level`.
- Scaled numeric features using `StandardScaler`.

### 2. Exploratory Data Analysis (EDA)
- Distribution plots for the target variable (`diabetes`) and numerical features.
- Pairplots to understand feature relationships.
- Correlation heatmaps to visualize feature correlations.

### 3. Model Building
Built and evaluated multiple models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine (SVM)
- Naive Bayes
- Gradient Boosting
- CatBoost

### 4. Model Evaluation
For each model:
- Calculated **Accuracy**, **F1 Score**, and generated **Classification Reports**.
- Visualized **Model Performance Comparison** through bar charts.
- Plotted **Confusion Matrix** for LightGBM (best-performing model).

---

## 📊 Model Performance Summary

| Model                 | Accuracy | F1 Score |
|-----------------------|----------|----------|
| LightGBM              | 97.13%   | 0.8100   |
| Gradient Boosting     | 97.10%   | 0.8062   |
| CatBoost              | 97.02%   | 0.8051   |
| XGBoost               | 96.98%   | 0.8012   |
| Random Forest         | 96.83%   | 0.7879   |
| SVM                   | 96.50%   | 0.7557   |
| K-Nearest Neighbors   | 95.73%   | 0.6827   |
| Decision Tree         | 94.73%   | 0.6771   |
| Logistic Regression   | 92.67%   | 0.5938   |
| Naive Bayes           | 90.01%   | 0.5273   |

📈 LightGBM achieved the best results based on both accuracy and F1 score!

---

## 📌 Technologies Used

- Python 🐍
- Pandas, NumPy (Data handling)
- Seaborn, Matplotlib (Visualization)
- Scikit-learn (Preprocessing, Models, Evaluation)
- XGBoost, LightGBM, CatBoost (Boosting Models)
- Flask

---

## 🚀 How to Run

1. Clone this repository.
2. Install dependencies:  
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost flask
   ```
3. Run the Python script or Jupyter notebook.

---

## 📬 Connect with Me

- LinkedIn: [Nivash](https://www.linkedin.com/in/nivash-r-n/)
- Portfolio: [Nivash](https://rnnivash.github.io/My_Port/)
- Email: [hello.nivashinsights@gmail.com](mailto:hello.nivashinsights@gmail.com)

🔍 **Let's leverage data science to make healthcare better!** 🚀
