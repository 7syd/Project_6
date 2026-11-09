# Rossmann Store Sales Forecasting

**Project 06 | Data Science Internship**

A machine learning project focused on predicting daily retail sales and customer traffic across Rossmann stores using historical sales data, store characteristics, promotions, holidays, and time-based features.

---

## 1. Project Overview

Rossmann operates thousands of retail stores across Europe. Store sales are influenced by several factors, including promotions, seasonal trends, holidays, customer traffic, and competition.

The objective of this project is to analyze historical retail data, develop predictive machine learning models, and build an interactive application that allows users to generate sales and customer forecasts for individual stores.

The project combines exploratory data analysis, feature engineering, machine learning, experiment tracking, data versioning, and application deployment.

## 2. Objectives

- Analyze historical sales and customer behaviour.
- Identify important factors influencing retail performance.
- Perform exploratory data analysis and visualization.
- Develop machine learning models for sales forecasting.
- Predict customer traffic using a separate regression model.
- Explore deep learning approaches using LSTM.
- Track experiments using MLflow.
- Manage dataset versions using DVC.
- Develop an interactive Streamlit forecasting dashboard.

## 3. Dataset Overview

The project uses the Rossmann Store Sales dataset.

| Dataset | Records |
|---|---:|
| Training dataset | 1,017,209 |
| Testing dataset | 41,088 |
| Store metadata | 1,115 stores |

**Date ranges:**
- Training: January 2013 – July 2015
- Testing: August 2015 – September 2015

**Important features:**

| Feature | Description |
|---|---|
| Sales | Daily store revenue |
| Customers | Daily customer count |
| Store | Unique store identifier |
| DayOfWeek | Day of the week |
| Promo | Promotion indicator |
| StateHoliday | State holiday indicator |
| SchoolHoliday | School holiday indicator |
| CompetitionDistance | Distance to nearest competitor |
| StoreType | Store classification |
| Assortment | Store assortment level |

## 4. Data Preparation & Feature Engineering

The following preprocessing steps were performed:

- Inspected missing values and dataset structure.
- Handled missing store metadata.
- Imputed missing store-opening information in the test dataset.
- Merged sales records with store metadata.
- Processed categorical variables.
- Created date-based features.
- Prepared chronological training and validation datasets.

**Engineered features included:**
- Year, Month, Day and WeekOfYear.
- DayOfWeek.
- IsWeekend.
- IsMonthStart and IsMonthEnd.
- Competition-related features.
- Promotion-related features.
- Store characteristics.

## 5. Exploratory Data Analysis

Exploratory analysis was conducted to understand retail sales patterns and customer behaviour.

Key observations:

- Promotions were associated with increased sales and customer traffic.
- Seasonal variations influenced average store sales.
- Customer counts showed a strong positive relationship with sales.
- Store characteristics and competition provided additional predictive information.

The correlation between Sales and Customers was approximately **0.895**.

Visual analysis included:
- Sales distribution.
- Monthly sales trends.
- Promotion versus non-promotion performance.
- Customer count analysis.
- Sales and customer correlation.
- Competition analysis.

## 6. Machine Learning Models

Multiple approaches were explored and evaluated.

**Models developed:**

1. Store Average Baseline Model
2. Random Forest Regressor
3. Extra Trees Regressor
4. Extra Trees Customer Forecasting Model
5. Global LSTM Time-Series Model

### Sales Model Evaluation

The sales models were evaluated using a chronological validation split, with training data before June 2015 and validation data from June 2015.

| Model | MAE | RMSE |
|---|---:|---:|
| Store Average Baseline | 1,486.65 | 1,979.55 |
| Random Forest | 710.71 | 1,056.97 |
| Extra Trees | **705.14** | **1,021.69** |

**Best-performing sales model: Extra Trees Regressor**

The Extra Trees model achieved approximately **52.6% lower MAE** compared with the store-average baseline.

### Customer Forecasting

A separate Extra Trees regression model was developed for customer forecasting.

| Metric | Result |
|---|---:|
| MAE | 55.56 |
| RMSE | 79.40 |

### LSTM Experiment

A global LSTM model was also developed to explore deep learning approaches for time-series forecasting.

The trained model and its preprocessing artifacts were saved for reproducibility and further experimentation.

## 7. Model Tracking & Data Versioning

### MLflow

MLflow was integrated to track machine learning experiments.

Implementation included:
- Experiment creation and tracking.
- Recording model training runs.
- Tracking model parameters and evaluation metrics.
- SQLite-backed experiment storage.
- Maintaining experiment history.

### DVC

Data Version Control was implemented to manage dataset versions.

Implementation included:
- DVC initialization.
- Tracking training, testing, and store datasets.
- Creating Git-tracked dataset metadata.
- Demonstrating dataset version changes through snapshots.

Git was used alongside DVC to maintain version history.

## 8. Streamlit Application

An interactive forecasting dashboard was developed using Streamlit.

**Application features:**

- Store selection.
- Custom forecast date ranges.
- Store opening status selection.
- Promotion configuration.
- School holiday configuration.
- State holiday configuration.
- Daily sales predictions.
- Daily customer predictions.
- Forecast summary metrics.
- Sales and customer trend visualizations.
- Prediction results table.
- CSV forecast download.

The application uses separately trained sales and customer models to generate forecasts based on the selected store and input conditions.

**Important:** Promotion and holiday selections apply uniformly across the selected forecast dates. Forecasts are estimates rather than actual sales figures.

## 9. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Machine learning |
| TensorFlow / Keras | LSTM development |
| Joblib | Model serialization |
| Streamlit | Interactive dashboard |
| MLflow | Experiment tracking |
| DVC | Dataset version control |
| Git | Source and version management |
| SQLite | MLflow backend storage |

## 10. Project Structure

```text
Proj_6/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── store.csv
│   └── *.dvc
│
├── Notebooks/
│   └── Project analysis and model development
│
├── models/
│   ├── best_extratrees_pipeline.pkl
│   ├── sales_app_bundle.pkl
│   ├── customer_extratrees_pipeline.pkl
│   ├── customer_app_bundle.pkl
│   ├── store_info.pkl
│   ├── global_lstm.keras
│   ├── lstm_scaler.pkl
│   └── lstm_preprocessing.pkl
│
├── outputs/
│   ├── rossmann_submission.csv
│   └── rossmann_mlflow_submission.csv
│
├── logs/
│
├── mlruns/
├── .dvc/
├── .dvcignore
├── mlflow.db
├── app.py
└── requirements.txt
```

## 11. Installation & Execution

### Prerequisites

- Python 3.10 or compatible environment.
- Git.
- Required project datasets and trained model artifacts.

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd Proj_6
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Launch the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will open in your default browser.

**Note:** Run the application from the project root directory so that the relative model and data paths resolve correctly.

## 12. Key Results & Achievements

- Processed more than one million training records.
- Identified important retail sales and customer behaviour patterns.
- Achieved approximately 52.6% lower validation MAE compared with the baseline.
- Developed separate sales and customer forecasting models.
- Explored LSTM-based time-series forecasting.
- Integrated MLflow for experiment tracking.
- Implemented DVC for dataset versioning.
- Developed a functional Streamlit forecasting dashboard.
- Generated test-set predictions and CSV submission files.

## 13. Future Improvements

Potential areas for further development include:

- Hyperparameter optimization.
- Improved time-series forecasting architectures.
- Automated model retraining.
- More granular holiday and promotion features.
- Integration with live retail datasets.
- Cloud deployment and API integration.
- Enhanced forecasting uncertainty estimation.

## 14. Conclusion

This project demonstrates an end-to-end retail forecasting workflow, covering data exploration, preprocessing, feature engineering, machine learning, deep learning experimentation, model tracking, dataset versioning, and interactive application development.

By combining predictive analytics with an accessible dashboard, the project demonstrates how historical retail data can be transformed into actionable insights to support business planning and decision-making.

---

**Project:** Rossmann Store Sales Forecasting  
**Domain:** Data Science | Retail Analytics | Machine Learning  
**Application:** Streamlit  
**Model:** Extra Trees Regressor  
**Experiment Tracking:** MLflow  
**Data Versioning:** DVC

**Author:** Your Name
