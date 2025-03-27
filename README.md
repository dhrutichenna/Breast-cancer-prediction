# Breast-cancer-prediction
# 🩺 Breast Cancer Detection using Machine Learning

## Overview
This project aims to detect breast cancer using machine learning techniques by classifying tumors as **benign** or **malignant** based on various diagnostic features. The model leverages real-world medical data to support early and accurate cancer detection, which can potentially save lives.

##  Model Used
- **Logistic Regression**: Chosen for its simplicity, interpretability, and high accuracy in binary classification problems.

##  Dataset Description
- **Source**: Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository
- **Size**: 569 rows × 33 columns
- **Features**: Includes attributes such as radius, texture, perimeter, area, smoothness, compactness, symmetry, etc.
- **Target**: Diagnosis (B = Benign, M = Malignant)

##  Key Steps in the Project

###  Data Preprocessing
- Handled missing values by dropping the `Unnamed: 32` column
- Encoded the target variable (`diagnosis`) using `LabelEncoder`
- Split the data into **training** and **testing** sets (80-20 split)
- Scaled features using **StandardScaler**

### Exploratory Data Analysis (EDA)
- Used `seaborn` and `matplotlib` for visualization
- Plotted the distribution of the target variable
- Created a correlation heatmap to observe relationships between features and diagnosis

###  Model Building and Evaluation
- Built a **Logistic Regression** classifier using `sklearn`
- Achieved an **accuracy of 97.36%** on the test set
- Used evaluation metrics like **confusion matrix** and **accuracy score**

###  Prediction System
- Developed a prediction function that takes user input (preprocessed) and classifies it as **Cancerous** or **Not Cancerous**

###  Model Deployment Ready
- Trained model saved using **Pickle** (`model.pkl`)
- Ready for integration into a deployment environment (e.g., Flask or Streamlit)

##  Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

##  My Contribution
- Performed all stages of development — data cleaning, visualization, model training, evaluation, and saving
- Wrote a mini prediction system to simulate real-world use
- Ensured reproducibility and clarity of the code


## 📬 Contact
Feel free to reach out for feedback or collaboration:
- 📧 Email: dhruticr@gmail.com
