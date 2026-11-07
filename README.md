
# GitHub Repository Analytics

An interactive analytics dashboard exploring 4.24 million public GitHub repositories using Apache Spark, Python, Streamlit and Plotly.

## Project Overview

This project analyses public GitHub repository metadata to understand repository popularity, programming languages, development activity, collaboration patterns and repository characteristics.

The original dataset was processed using PySpark, cleaned, deduplicated and converted into analytical outputs for visualization.

## Dataset Statistics

| Metric | Result |
|---|---:|
| Total repositories | 4,242,532 |
| Total stars | 412.9 million |
| Median stars | 12 |
| Top 1% share of all stars | 55.3% |
| Recently updated repositories | 23.6% |

## 🚀 Live Dashboard

Explore the interactive dashboard here:

👉 https://repo-insights-dashboard.streamlit.app

## Dashboard Features

- Repository Popularity Analysis
- Programming Language Analysis
- Repository Activity and Creation Cohorts
- Repository Health Analysis
- Collaboration Analysis
- GitHub Topics Analysis
- Correlation Analysis
- Machine Learning Results
- Interactive Data Explorer

## Machine Learning

The project uses:

- Logistic Regression
- Gradient-Boosted Trees (GBT)

The models classify repositories with at least 100 stars.

The project compares models using precision, recall, F1-score, accuracy, PR-AUC and ROC-AUC.

**Important:** The models classify current popularity, not future popularity. Models incorporating forks and watchers have a partly circular advantage because these variables are themselves indicators of popularity.

## Technology Stack

- Python
- Apache PySpark
- Pandas
- Streamlit
- Plotly
- Machine Learning
- Kaggle

## Run Locally

Clone the repository:

```bash
git clone https://github.com/anugopalv/github-repository-analytics.git
```

Navigate into the project:

```bash
cd github-repository-analytics
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

## Dataset Snapshot

The dataset represents approximately 13 September 2025.

Repository activity measurements are relative to this snapshot.

## Project Structure

```text
github-repository-analytics/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── analytical CSV files
```

## Author

Developed as a Big Data Analytics project using PySpark and Streamlit.
