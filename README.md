# Telco-Customer-Churn-Prediction

## 📌 Project Overview
Customer churn is one of the biggest challenges in the telecom industry. This project uses the **IBM Telco Customer Churn dataset** to predict whether a customer will leave ("Yes") or stay ("No").  

The goal is simple: **turn raw customer data into actionable insights** using machine learning.

---

## ⚙️ Workflow Stages
1. **Data Loading & Cleaning**  
   - Imported dataset from local `data/` folder  
   - Handled missing values and ensured consistency  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized churn distribution  
   - Explored relationships between features 

3. **Feature Engineering**  
   - Separated numeric and categorical features  
   - Prepared preprocessing pipelines  

4. **Model Building**  
   - Train/test split with stratification  
   - Random Forest Classifier inside a pipeline (imputation, scaling, one-hot encoding)  

5. **Evaluation**  
   - Accuracy, precision, recall, F1-score  
   - Confusion matrix visualization  

---

## 📊 Results
| Metric        | Value |
|---------------|-------|
| Accuracy      | 0.79  |
| Precision     | 0.65  |
| Recall        | 0.48  |
| F1-score      | 0.56  |
| ROC-AUC       | 0.82  |

*(Values based on confusion matrix: TN=939, FP=96, FN=193, TP=181)*

---

## 🛠️ Requirements
Dependencies are listed in `requirements.txt`:
