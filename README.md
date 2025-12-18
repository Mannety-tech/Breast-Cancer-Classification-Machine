🩺 Breast Cancer Classification Machine
A machine‑learning system for predicting benign vs malignant breast tumours using clinical measurements.

📌 Problem Statement
Breast cancer diagnosis relies heavily on clinical expertise, imaging interpretation, and laboratory measurements. However, early‑stage tumours often present subtle patterns that are difficult to detect consistently, leading to delayed diagnosis and poorer survival outcomes. Machine‑learning models can support clinicians by identifying complex, non‑linear patterns in medical data that may not be immediately visible through traditional examination.
This project investigates whether a reliable and reproducible machine‑learning model can be developed to predict tumour malignancy using structured clinical measurements. A secondary challenge explored is whether multiple heterogeneous breast‑cancer datasets can be combined into a unified dataset, and whether such integration improves predictive performance. The final solution must be accurate, interpretable, and deployable as a real‑time web application to support clinical decision‑making.

📘 Introduction
Breast cancer remains one of the most significant global health challenges, with early detection being crucial for improving patient outcomes. Machine learning has become an increasingly valuable tool for supporting clinical decision‑making by uncovering patterns in complex medical datasets that may not be immediately visible to clinicians.
This project aims to develop a unified machine‑learning model capable of predicting breast cancer malignancy by exploring three heterogeneous datasets:
A clinical dataset
A diagnostic imaging dataset
A third dataset initially labelled as a recurrence dataset, later identified as containing diagnostic‑style features
The project demonstrates a complete end‑to‑end ML workflow, including exploratory data analysis (EDA), data cleaning, schema alignment, preprocessing, model training, evaluation, and deployment.

📊 Dataset Exploration
Three datasets were analysed during the early stages of the project. The initial goal was to merge them into a single unified dataset. This required:
Schema alignment
Imputation
Feature engineering
Exploratory data analysis
Cleaning and validation

❗ Why the datasets could not be merged
During evaluation, it became clear that the datasets were not directly compatible. They differed in:
Feature definitions
Measurement scales
Clinical purpose
Merging them introduced inconsistencies that reduced model reliability and interpretability.

✅ Final Dataset Selection
To ensure scientific validity and reproducibility, the final predictive model was trained exclusively on the Breast Cancer Wisconsin Diagnostic Dataset, which provides:
30 consistent numerical tumour‑measurement features
No missing values
A well‑defined binary target (Malignant vs Benign)
Strong suitability for supervised ML classification
The combined dataset remains part of the project to demonstrate data exploration and integration attempts, but the deployed model uses only the diagnostic dataset.

🧹 Data Cleaning & Preprocessing
The following steps were applied:
Removed the id column (non‑predictive identifier)
Extracted diagnosis as the target variable
Standardised all 30 numerical features using StandardScaler
No imputation required (dataset contains no missing values)
This ensures a clean, reproducible, and clinically meaningful dataset for model training.

🎯 Project Objectives
Build a robust ML model to classify tumours as Benign (B) or Malignant (M)
Evaluate performance using clinically relevant metrics
Provide visual insights into model behaviour
Deploy the model through a simple, user‑friendly web interface

🧠 Model Choice: Why Random Forest?
Random Forest was selected because it is well‑suited to medical diagnostic tasks where patterns are often non‑linear and complex. It:
Handles high‑dimensional data effectively
Is robust to noise and outliers
Reduces overfitting through bootstrap aggregation
Provides interpretable feature importance
This makes it a reliable and clinically appropriate choice for breast cancer prediction.

📈 Evaluation Metrics & Results
✔ Accuracy
The model achieved 96% accuracy, demonstrating strong predictive performance.
✔ Confusion Matrix Interpretation
The confusion matrix shows that the model correctly identifies the majority of benign and malignant cases. It achieves high recall for malignant tumours, meaning it rarely misses dangerous cancer cases — a critical requirement in clinical settings.

✔ ROC Curve Interpretation
The ROC curve achieved an AUC of 0.98, indicating excellent separability between benign and malignant tumours. This reflects strong diagnostic power across different classification thresholds.

✔ Feature Importance Analysis
Key influential features include:
radius_worst
perimeter_worst
concave points_worst
These features align with clinical research showing that tumour size and shape irregularities are strong indicators of malignancy.

Technologies Used
Python
Pandas, NumPy
Scikit‑learn
Matplotlib
Flask
HTML/CSS

Project Structure
Code
Breast Cancer Classification Machine/
│
├── app.py                               # Flask application
├── breast_cancer_unified_model.pkl       # Final trained ML model
├── combined_breast_cancer_validated.csv  # Cleaned combined dataset (EDA only)
├── Software Artefact.ipynb               # Main ML notebook
├── cleaned dataset.ipynb                 # Dataset preparation notebook
├── templates/
│   └── index.html                        # Frontend UI
├── static/
│   └── style.css                         # Optional CSS assets
└── README.md                             # Project documentation


Machine Learning Model
Model
RandomForestClassifier
Wrapped in a scikit‑learn Pipeline
Preprocessing
Numeric features:
Median imputation
Standard scaling
Categorical features:
Most‑frequent imputation
One‑hot encoding

Flask Web Application
Endpoints
Route	Method	Description
/	GET	Loads the prediction form
/predict	POST	Processes user input and returns prediction

Prediction Flow
User enters tumour measurements
Inputs converted to numeric values
Feature vector aligned to model’s training order
Missing values replaced with 0
Model predicts Benign or Malignant
Optional confidence score displayed
Logging
All predictions are logged in:
Code
Prediction_logs.txt
Each entry includes:
Timestamp
User inputs
Model prediction

Known Limitations
UI currently supports a subset of all dataset features
Missing features are filled with 0 (safe for RandomForest but not ideal for all models)
Local‑only deployment unless hosted on a cloud platform

Future Improvements
Add full feature support in the UI
Add SHAP explainability
Deploy to Render / Azure / Heroku
Add authentication
Add probability visualisation
Review confidence score

Environment Setup (Conda – Python 3.7)
Clone the repository
Code
git clone <repo-url>
cd "Breast Cancer Classification Machine"
Create the environment
Code
conda create -n py37 python=3.7
conda activate py37
Install dependencies
Code
pip install flask pandas scikit-learn joblib
pip install -r requirements.txt
Run the Flask app
Code
python app.py
App runs at:
http://127.0.0.1:5001

Run Jupyter Notebook
Code
conda activate py37
jupyter notebook

 Conclusion
This project demonstrates a complete machine‑learning workflow for breast cancer classification, combining strong predictive performance with clear interpretability. The system is reliable, clinically relevant, and ready for further development or deployment.

👤 Author
Mannety Mowa  
BSc (Hons) Computer Science
Leeds Trinity University


