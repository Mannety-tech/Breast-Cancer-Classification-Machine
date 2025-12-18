Problem Statement 

Breast cancer diagnosis relies heavily on clinical expertise, imaging interpretation, and laboratory measurements. However, earlystage tumours often present subtle patterns that are difficult to detect consistently, leading to delayed diagnosis and reduced survival outcomes. Machine-learning models offer the potential to support clinicians by identifying complex, non-linear patterns in medical data that may not be immediately visible through traditional examination. 

The project addresses the challenge of building a reliable and reproducible machine-learning model capable of predicting breast tumour malignancy using structured clinical measuarements. An additional challenge explored in this work is whether multiple heterogeneous breast-cancer datasets can be combined into a unified dataset, and whether such integration improves predictive performance. The final solution must be accurate, interpretable, and deployable as a real-time web application to support clinical decision-making. 


Introduction 


Breast cancer remains one of the most significant global health challenges, with early detection playing a critical role in improving patient outcomes. Machine learning has become an increasingly valuable tool for supporting clinical decision-making by identifying patterns in complex medical datasets that may not be immediately visible to clinician. This project aims to develop a unified machine-learning model capable of predicting breast cancer malignancy by integrating three heterogeneous datasets: a clinical dataset, a diagnostic imaging dataset, and a third dataset initially labelled as a “recurrence” but later identified as containing diagnostic features. 

The primary objective was to explore whether combining clinical and diagnostic information can improve predictive performance compared to using a single dataset. This required careful data cleaning, schema alignment, imputation, feature engineering, and model evaluation. The project demonstrates a full end-to-end machine-learning pipeline, including exploratory data analysis (EDA), preprocessing, model training, evaluation, and reflection

Dataset Exploration 

Three datases were explored during the early stages of the project: 

A clinical dataset 

A diagnostic imaging dataset 

A third dataset initially labelled as a recurrence dataset, later identified as containing diagnostic-style features 

The initial goal was to combine these datasets into a single unified dataset. This required: 

Schema alignment  

Imputation 

Feature engineering  

Exploratory data analysis (EDA) 

Cleaning and Validation 

However, during evaluation it became clear that the datasets were not directly compatible. They differed in: 

Feature definitions 

Measurement scales 

Clinical purpose 

Merging them introduced inconsistencies that reduced model reliability. 

 

Final Dataset Selection 

For these reasons, the final predictive model was trained exclusively on the Breast Cancer Wisconsin Diagnostic dataset, which offers: 

A complete and consistent set of 30 numerical tumour-measurement features 

No missing values 

A well-defined binary target (diagnosis: Malignant vs Benign) 

Strong suitability for supervised machine-learning classification 

The combined dataset remains part of the project to demonstrate data exploration, cleaning, and integration attempts, but deployed model uses only the diagnostic dataset to ensure scientific validity and reproducibility. 

Data Cleaning and Preprocessing Steps 

The id column was removed because it is an identifier and carries no predictive value. 

The diagnosis column was separated as the target variable. 

All 30 numerical features were standardised using StandardScaler to ensure equal weighing during model training. 

No imputation or outlier removal was required, as the dataset contains no missing or invalid values. 

By using the Diagnostic dataset alone, the final model is trained on high-quality, clinically meaningful features, ensuring reproducibility, interpretability, and alignment with established research standards. 

The Breast Cancer Classification Machine is a machine-learning powered web application that predicts whether a breast tumour is Benign or Malignant based on clinical measurements. It demonstrates practical ML deployment using: 

A RandomForestClassifier 

A scikit-learn preprocessing pipeline 

A Flask web interface 

A clean, user-friendly HTML form  

Automatic feature alignment for safe predictions 

The project was developed as part of a Computer Science degree at Leeds Trinity University. 

Features 

Accurate tumour classification (Benign vs Malignant) 

Flask-based web interface 

Automatic preprocessing (imputation, scaling, encoding) 

Feature-alignment layer to prevent KeyErrors 

Optional confidence scores 

Prediction logging for audit and reporting 

Bootstrap-styled UI 

 

 

 

 Project Structure 

Code 

Breast Cancer Classification Machine/
│
├── app.py                               # Flask application
├── breast_cancer_unified_model.pkl       # Final trained ML model
├── combined_breast_cancer_validated.csv  # Cleaned combined dataset (EDA only)
├── templates/
│   └── index.html                        # Frontend UI
├── static/                               # Optional CSS/JS assets
└── README.md                             # Project documentation


 

Machine Learning Model 

 

Model  

RandomForestClassifier 

Wrapped in a scikit-learn Pipeline 

 

Preprocessing  

Numeric features 

Median imputation 

Standard scaling 


Categorical features 

Most-frequent imputation 

One-hot encoding 

 

Training Environment 

 

Python 3.7 

Scikit-learn 1.0.2 

Joblib protocol=4 (ensures compatibility with Flask) 

 

 

Flask Application 

 

Endpoints 

 

Route                      Method                 Description 

/                                 GET                       Loads the prediction form 

 

/predict                  POST                    Processes user input and returns predict ion 

 

Prediction Flow 

1.User enters tumour measurements  

2. Inputs are safely converted to numeric values (floats) 

3.A complete feature vector is constructed to match the model’s training feature order 

4. Any missing or empty inputs are replaced with 0 

5. The model predicts whether the tumour is Benign or Malignant 

6 (Optional) A confidence score is generated and displayed to the user 

 

 
Usage 

Open the web app in your browser 

Enter tumour measurements 

Click Predict 

View the classification result 

(Optional) Review confidence score 

Logging 

 

All predictions are logged to: 

 

Code 

Prediction_logs.txt 

 

Each entry includes: 

Timestamp 

User inputs 

Model prediction 

 

Perfect for reports and debugging. 

 

Known Limitations 

 

UI currently supports a subset of all dataset features 

Missing features are filled with 0 (safe for RandomForest but not ideal for all models) 

Local-only deployment unless hosted on a cloud platform 

 


Future Improvements  

Add full feature support in the UI 

Add SHAP explainability 

Deploy to Render/ Azure/ Heroku 

Add authentication 

Add probability visualisation (progress bar or gauge) 



Environment Setup (conda- Python 3.7) 

This project uses a dedicated conda environment named py37 to ensure compatibility with the machine-learning model and Flask application. 

Clone the repository 


bash 

Git clone < > 

cd “Breast Cancer Classification Machine” 


How to create the environment 

conda create -n py37 python=3.7 

Activate the environment  

conda activate py37 

When activated the terminal will prompt  

(py37) mannetymowa@Mac Breast Cancer Classification Machine % 

Install project dependencies  

bash 

pip install flask pandas scikit-learn joblib 

pip install -r requirements.txt 


Run the Flask app 

 

bash 

python app.py 

 

The app will run at: 

Running on 

 http://127.0.0.1:5001 

How to run jupyter notebook on the environment if not installed 

pip install jupyter

Run:
conda activate py37
jupyter notebook
http://localhost:8888/tree



 

Author 

 

Mannety Mowa 

BSc (Hons) Computer Science  

Leeds Trinity University


