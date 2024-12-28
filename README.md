# Stroke Prediction Using Machine Learning

This repository contains a Machine Learning project developed to predict the likelihood of a patient experiencing a stroke. The project was completed as part of the **Data Science Engineering** course offered by **Great Learning**.

## Project Overview
This project explores a dataset with various medical attributes to predict stroke occurrences using Machine Learning techniques. The analysis includes data preprocessing, exploratory data analysis, feature engineering, and model selection to achieve accurate predictions.

---

## Dataset Description

- **Id**: Unique identifier
- **Gender**: Gender of the patient
- **Age**: Age of the patient
- **Hypertension**: Binary (0: No, 1: Yes)
- **Heart Disease**: Binary (0: No, 1: Yes)
- **Ever Married**: Binary (0: No, 1: Yes)
- **Work Type**: Categorical (e.g., Private, Self-employed, Govt-job)
- **Residence Type**: Urban/Rural
- **Average Glucose Level**: Numeric
- **BMI**: Body Mass Index
- **Smoking Status**: Categorical (e.g., Never smoked, Smokes)
- **Stroke**: Binary (Target variable; 0: No, 1: Yes)

## Project Workflow

1. **Data Analysis and Preprocessing**:
   - Check for missing values and handle them appropriately.
   - Explore data distributions and relationships using visualizations.
   - Encode categorical variables and standardize numerical features.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize the distribution of key features such as BMI, glucose levels, and age.
   - Analyze the proportion of stroke occurrences in the dataset.
   - Investigate correlations using a heatmap.

3. **Feature Engineering**:
   - Transform categorical variables into integer formats.
   - Handle class imbalance using techniques like oversampling or undersampling.

4. **Model Building and Evaluation**:
   - Experiment with various models such as Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting.
   - Perform hyperparameter tuning to optimize model performance.
   - Evaluate models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

## Results
The final model was selected based on its ability to balance sensitivity and specificity, making it suitable for medical predictions. The repository includes the evaluation metrics and a detailed comparison of model performances.

## Files in This Repository
- **`Stroke Prediction.ipynb`**: Jupyter notebook containing the entire analysis and modeling process.
- **Dataset**: The dataset used for this project.

## Prerequisites
Ensure the following dependencies are installed:

- Python 3.9
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run the Code
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/stroke-prediction.git
   ```
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook Stroke\ Prediction.ipynb
   ```
3. Follow the steps in the notebook to reproduce the results.

## Acknowledgments
This project was completed as part of the **Data Science Engineering** course by **Great Learning**. Special thanks to the instructors and peers for their guidance and support.

## Contact
For any queries or feedback, feel free to reach out:
- **Name**: Basil John Milton Muthuraj
- **Email**: bjmm1296@gmail.com
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/bjmm1296)

---
