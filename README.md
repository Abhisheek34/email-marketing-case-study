# email-marketing-case-study
Optimizing an e‑commerce email campaign with machine learning to predict and boost click‑through rates. Includes data processing, EDA, XGBoost modeling, performance evaluation, and actionable business insights.

# Final Report

Objective:
The primary goal of this case study is to evaluate the performance of an email marketing campaign and build a machine learning model to improve its effectiveness, particularly the click-through rate (CTR).

Business Context:
An e-commerce platform launched an email campaign to inform users about a new feature. The company randomly selected users and sent them an email. Success was defined as users clicking on a link inside the email. Our task was to:

Analyze how the campaign performed.
Build a machine learning model to increase the likelihood of users clicking the email.
Estimate how much the model could improve the CTR.
Identify any interesting user segment patterns.

## Data Preprocessing

### 1. Data Merging and Feature Engineering
- Merged `email_table`, `email_opened_table`, and `link_clicked_table` on `email_id`.
- Created binary flags `opened` and `clicked`.
- One‑hot encoded categorical variables (`weekday`, `user_country`).
- Label‑encoded binary features (`email_text`, `email_version`).

### 2. Handling Class Imbalance
- The target (`clicked`) was highly imbalanced (<1% positives).
- Applied **oversampling** of the minority class (clicked = 1) to balance the training data, ensuring the model learns click patterns effectively.

### 3. Feature Scaling
- No scaling was applied for XGBoost (tree‑based model).
- For linear or distance‑based models, standard scaling would be recommended.

## Model Performance Comparison

| Model                | Train Accuracy | Test Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | AUC Score |
|----------------------|----------------|---------------|---------------|---------------|------------|------------|--------------|--------------|-----------|
| **Logistic Regression** | 0.68           | 0.68          | 0.67          | 0.68          | 0.69       | 0.67       | 0.68         | 0.68         | 0.74      |
| **XGBoost**           | 0.92           | 0.92          | 1.00          | 0.87          | 0.85       | 1.00       | 0.92         | 0.93         | 0.92      |
| **ANN (Sequential)**  | 0.70           | 0.70          | 0.74          | 0.67          | 0.61       | 0.78       | 0.67         | 0.72         | 0.69      |

### Notes:
- **Logistic Regression**: A traditional approach, giving us a baseline. Although not the best at classification, it shows reasonable performance on the dataset.
- **XGBoost**: The best-performing model, achieving high accuracy, precision, and recall, especially for the minority class. It provides a balanced tradeoff between the precision and recall of both classes.
- **ANN (Sequential)**: The neural network shows decent results, but is more prone to underperforming due to overfitting on the majority class, even though it handles the minority class better than Logistic Regression.


# Results
XGBoost Model Performance:
Train Accuracy: 92.5%
Test Accuracy: 92.2%
Precision (Class 0): 1.00
Precision (Class 1): 0.87
Recall (Class 0): 0.85
Recall (Class 1): 1.00
F1-Score: 0.93 (both classes)

# Key Findings:
The XGBoost model achieved a high accuracy and ROC-AUC score, making it the best model for predicting email link clicks.

Personalized emails and emails sent during certain hours (e.g., late morning) showed better engagement rates.

Users who have made more purchases in the past were more likely to click on the email link.
