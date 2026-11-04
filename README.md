# Retail-Demand-Intelligence-Predictive-Inventory-Analytics

### Retail Demand Forecasting & Predictive Inventory Analytics

Demand Intelligence is an end-to-end retail analytics and
demand-forecasting application built using real point-of-sale
transaction data from a retail business. It processes sales records,
performs SQL-based analysis, compares machine-learning regression
models, and presents sales insights, demand forecasts, and inventory
replenishment estimates through an interactive Streamlit dashboard.

> **Project scope:** This is a decision-support prototype. Forecasts and
> replenishment suggestions are estimates, not live inventory or
> procurement instructions.

## Key Features

-   **Sales data preparation:** inspect, validate, clean, and transform
    transaction records using Python and Pandas.
-   **Data quality checks:** check missing values, duplicate records,
    invalid quantities, and revenue consistency.
-   **SQL analytics:** store and query cleaned sales records using
    SQLite.
-   **Exploratory data analysis:** analyse product performance, revenue
    trends, daily sales, and weekday/weekend patterns.
-   **Demand forecasting:** prepare time-based and lag-based features
    and train three regression models.
-   **Model comparison:** evaluate Linear Regression, Random Forest, and
    Gradient Boosting against a seven-day moving-average baseline.
-   **Interactive dashboard:** explore sales, filter records, view
    forecasts, and inspect model performance using Streamlit and Plotly.
-   **Inventory planning:** estimate replenishment quantities from
    forecast demand, a manually entered stock level, and a configurable
    safety-stock buffer.
-   **Evaluation diagnostics:** inspect actual-versus-predicted demand,
    prediction errors, and Linear Regression coefficients.

## Technology Stack

  Area                Technologies
  ------------------- ----------------------------------------
  Programming         Python
  Data processing     Pandas, NumPy
  Database / SQL      SQLite, Python `sqlite3`
  Machine learning    Scikit-learn
  Visualisation       Matplotlib, Seaborn, Plotly
  Dashboard           Streamlit
  Model persistence   Joblib
  Development         Jupyter Notebook, VS Code, Git, GitHub

## Dataset

The project uses six months of real retail POS transaction data covering
**1 March to 31 August 2026**.

The source workbook contains transaction-level fields including:

-   Date
-   Product
-   Variant / Pack Size
-   Unit Price
-   Quantity Sold
-   Revenue

### Dataset summary

  Measure                                                 Value
  --------------------------------- ---------------------------
  Cleaned POS transaction records                         4,008
  Products                                                   48
  Product variants / pack sizes                              66
  Units sold                                              5,593
  Total revenue                                      ₹14,73,533
  Prepared modelling rows                                11,504
  Modelling date range                8 March -- 31 August 2026

The prepared modelling dataset has more rows than the transaction
dataset because it is structured at the product/variant and date level
for demand modelling.

**Data privacy:** The original workbook, local SQLite database,
generated CSV datasets, and trained model files are excluded from this
public repository. The source retail data is not redistributed. You must
have permission to use any business data you provide locally.

## Project Workflow

``` text
POS transaction workbook
        |
        v
Data inspection and quality validation
        |
        v
Cleaning and preparation with Pandas
        |
        +----------------------+
        |                      |
        v                      v
SQLite database          Exploratory analysis
        |                      |
        v                      v
SQL analysis             Charts and insights
        |
        v
Demand feature preparation
        |
        v
Chronological train/test split
        |
        v
Linear Regression | Random Forest | Gradient Boosting
        |
        v
Evaluation against 7-day moving-average baseline
        |
        v
Saved model, predictions and evaluation outputs
        |
        v
Streamlit + Plotly dashboard
        |
        +--> Sales analytics
        +--> Seven-day demand forecasts
        +--> Inventory restock estimates
        +--> Model performance and diagnostics
```

## Machine-Learning Approach

The target variable is **Quantity Sold**. The model uses product and
variant identifiers alongside calendar and historical-demand features.

### Features used

-   Product
-   Variant / Pack Size
-   Day
-   Month
-   Day of week
-   Weekend indicator
-   One-day demand lag
-   Seven-day demand lag
-   Seven-day rolling mean

Categorical fields are one-hot encoded. The data is split
chronologically, with earlier dates used for training and later dates
used for testing.

### Models compared

1.  Linear Regression
2.  Random Forest Regressor
3.  Gradient Boosting Regressor
4.  Seven-day moving-average baseline

### Recorded test results

  Model                                      MAE           RMSE
  ------------------------------- -------------- --------------
  **Linear Regression**             **0.552606**   **0.708446**
  Gradient Boosting                     0.565735       0.710579
  Random Forest                         0.579050       0.730734
  7-Day Moving Average Baseline         0.557239       0.754678

On this test split, Linear Regression achieved the lowest MAE and RMSE
among the three trained regression models and also improved on the
moving-average baseline.

-   **MAE** measures the average absolute difference between actual and
    predicted demand.
-   **RMSE** penalises larger prediction errors more heavily than MAE.

These scores describe the saved test evaluation. They do not guarantee
the same accuracy for future dates, products, or businesses.

## Dashboard Modules

The Streamlit application contains six sections:

1.  **Sales Overview** --- revenue, units sold, transaction count,
    monthly revenue, and leading products.
2.  **Sales Analytics** --- date and product filters, daily revenue
    trends, product performance tables, and CSV export.
3.  **Demand Forecasting** --- recursive seven-day demand estimates for
    a selected product and variant.
4.  **Restock Planning** --- indicative order quantity based on forecast
    demand, manually entered available stock, and a safety-stock
    percentage.
5.  **Model Performance** --- MAE and RMSE comparison across the trained
    models and baseline.
6.  **Model Diagnostics** --- actual-versus-predicted charts, scatter
    plot, residual distribution, coefficient analysis, and prediction
    export.

## Repository Structure

``` text
Demand-Intelligence/
├── data/
│   ├── raw/                 # Local source workbook; not committed
│   ├── processed/           # Generated data/model files; not committed
│   └── retail_sales.db      # Local SQLite database; not committed
├── app.py
├── inspect_data.py
├── data_quality.py
├── clean_data.py
├── eda.py
├── create_database.py
├── sql_analysis.py
├── sql_pandas_analysis.py
├── prepare_model_data.py
├── train_models.py
├── model_evaluation.py
├── requirements.txt
└── .gitignore
```

## Local Setup

### 1. Clone the repository

``` bash
git clone https://github.com/sachinbabu-git/Retail-Demand-Intelligence-Predictive-Inventory-Analytics.git
cd Retail-Demand-Intelligence-Predictive-Inventory-Analytics
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` powershell
pip install -r requirements.txt
```

### 4. Add the source dataset

Create the following directory if it does not exist:

``` text
data/raw/
```

Place your authorised source workbook there as:

``` text
data/raw/sales_data.xlsx
```

The source workbook is intentionally not included in this public
repository.

### 5. Generate the local data and model artifacts

Run the scripts from the repository root, in this order:

``` powershell
python inspect_data.py
python data_quality.py
python clean_data.py
python eda.py
python create_database.py
python sql_analysis.py
python sql_pandas_analysis.py
python prepare_model_data.py
python train_models.py
python model_evaluation.py
```

These scripts generate the local processed datasets, SQLite database,
trained model, predictions, and evaluation files required by the
dashboard.

### 6. Launch the dashboard

``` powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually
`http://localhost:8501`.

## Limitations and Future Improvements

-   The available dataset covers six months; longer history is needed to
    assess annual seasonality and longer-term patterns.
-   Model performance should be validated with additional rolling or
    walk-forward time-based evaluation.
-   Forecast quality should be examined separately for individual
    products and variants.
-   The seven-day dashboard forecast is recursive: each predicted day is
    used to construct lag features for the following day, so errors can
    accumulate.
-   The restock planner uses manually entered stock and a configurable
    safety buffer. It does not currently model supplier lead times,
    purchase constraints, live inventory movements, or procurement
    costs.
-   Future work could include lead-time-aware reorder points,
    product-level evaluation reports, automated data refresh, and
    deployment.

## Author

**Sachin**

Built as a practical retail analytics and demand-forecasting project
using real business transaction data.

------------------------------------------------------------------------

If you use this project as a reference, adapt the data pipeline and
validation strategy to the structure and operational requirements of
your own retail dataset.
