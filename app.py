
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GitHub Repository Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

st.title("📊 GitHub Repository Analytics")
st.caption(
    "Big Data Analytics using PySpark | "
    "4.24 Million Public GitHub Repositories"
)

# ============================================================
# 2. DATA LOADING
# ============================================================

@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        return None

    return pd.read_csv(path)


def get_data(filename):
    df = load_csv(filename)

    if df is None:
        st.warning(f"Dataset unavailable: {filename}")

    return df


def pct(value):
    return f"{value * 100:.1f}%"


def show_chart(fig, height=430):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=65, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)


csv_files = sorted(DATA_DIR.glob("*.csv"))

# ============================================================
# 3. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Analytics Navigation")

page = st.sidebar.radio(
    "Explore the project",
    [
        "Overview",
        "Repository Popularity",
        "Programming Languages",
        "Activity & Cohorts",
        "Repository Health",
        "Collaboration Analysis",
        "Topics Analysis",
        "Correlation Analysis",
        "Machine Learning Results",
        "Data Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.metric("Analytical datasets", len(csv_files))

st.sidebar.caption(
    "Data processed using Apache Spark"
)

# ============================================================
# 4. OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Project Overview")

    overview = get_data("overview.csv")
    percentiles = get_data("stars_percentiles.csv")

    if overview is None or overview.empty:
        st.error("Overview data is unavailable.")
        st.stop()

    summary = overview.iloc[0]

    if percentiles is None or percentiles.empty:
        st.error("Percentile data is unavailable.")
        st.stop()

    median_row = percentiles[
        percentiles["percentile"].round(2) == 0.50
    ]

    if median_row.empty:
        st.error("Median stars value is unavailable.")
        st.stop()

    median_stars = int(median_row.iloc[0]["stars"])

    # Main KPIs
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Repositories", f"{int(summary['repos']):,}")
    c2.metric("Total Stars", f"{summary['total_stars']/1_000_000:.1f}M")
    c3.metric("Median Stars", f"{median_stars:,}")
    c4.metric(
        "Top 1% Star Share",
        pct(summary["share_of_all_stars_at_or_above_p99"])
    )

    st.divider()

    st.subheader("Repository Characteristics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Archived", pct(summary["share_archived"]))
    c2.metric("Licensed", pct(summary["share_licensed"]))
    c3.metric(
        "Recently Updated",
        pct(summary["share_active_365d_of_valid_dates"])
    )
    c4.metric("With Topics", pct(summary["share_with_topics"]))

    st.caption(
        "Recently updated means pushed within 365 days before "
        "the dataset snapshot, among repositories with valid dates."
    )

    st.divider()

    # Popularity chart
    st.subheader("Repository Star Distribution")

    stars = get_data("stars_log_hist.csv")

    if stars is not None:
        fig = px.bar(
            stars,
            x="log10_stars_lower",
            y="count",
            title="Distribution of Repository Stars",
            labels={
                "log10_stars_lower": "Log10 Stars - Lower Bin",
                "count": "Repositories"
            }
        )
        show_chart(fig)

    # Language chart
    st.subheader("Top Programming Languages")

    languages = get_data("language_summary.csv")

    if languages is not None:
        top = languages.nlargest(10, "repos").sort_values("repos")

        fig = px.bar(
            top,
            x="repos",
            y="primary_language_clean",
            orientation="h",
            title="Top 10 Languages by Repository Count",
            labels={
                "repos": "Repositories",
                "primary_language_clean": "Language"
            }
        )
        show_chart(fig)

    # Activity chart
    st.subheader("Repository Activity")

    recency = get_data("recency_buckets.csv")

    if recency is not None:
        recency = recency.sort_values("bucket_order")

        fig = px.bar(
            recency,
            x="bucket",
            y="repos",
            title="Repositories by Last Push Activity",
            labels={
                "bucket": "Time Since Last Push",
                "repos": "Repositories"
            }
        )
        show_chart(fig)

    st.divider()

    st.subheader("Project Methodology")

    st.markdown("""
    This project analyses public GitHub repository metadata using
    Apache Spark and machine learning.

    **Data engineering**
    - JSON dataset conversion and preparation.
    - PySpark data processing and cleaning.
    - Duplicate repository investigation and removal.
    - Parquet storage for analytical processing.

    **Machine learning**
    - Logistic Regression.
    - Gradient-Boosted Trees.
    - Classification of repositories with at least 100 stars.
    """)

    st.info(
        "The models classify current popularity, not future popularity. "
        "Including forks and watchers as features introduces circularity "
        "because these metrics are themselves indicators of popularity."
    )

    st.caption(
        "Dataset snapshot: approximately 13 September 2025."
    )


# ============================================================
# 5. REPOSITORY POPULARITY
# ============================================================

elif page == "Repository Popularity":

    st.header("⭐ Repository Popularity Analysis")

    stars = get_data("stars_log_hist.csv")
    percentiles = get_data("stars_percentiles.csv")
    overview = get_data("overview.csv")

    if stars is not None:

        fig = px.bar(
            stars,
            x="log10_stars_lower",
            y="count",
            title="Repository Star Distribution",
            labels={
                "log10_stars_lower": "Log10 Stars - Lower Bin",
                "count": "Number of Repositories"
            }
        )
        show_chart(fig, 500)

    if percentiles is not None:

        st.subheader("Star Count Percentiles")

        fig = px.line(
            percentiles,
            x="percentile",
            y="stars",
            markers=True,
            title="Repository Stars Across Percentiles",
            labels={
                "percentile": "Percentile",
                "stars": "Stars"
            }
        )
        show_chart(fig)

        st.dataframe(
            percentiles,
            use_container_width=True,
            hide_index=True
        )

    if overview is not None:

        row = overview.iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "P99 Star Threshold",
            f"{int(row['p99_stars_threshold']):,}"
        )

        c2.metric(
            "Repositories at P99 or Above",
            f"{int(row['repos_at_or_above_p99']):,}"
        )

        c3.metric(
            "Share of All Stars Held by Top 1%",
            pct(row["share_of_all_stars_at_or_above_p99"])
        )


# ============================================================
# 6. PROGRAMMING LANGUAGES
# ============================================================

elif page == "Programming Languages":

    st.header("💻 Programming Language Analysis")

    languages = get_data("language_summary.csv")
    language_bytes = get_data("language_bytes.csv")

    if languages is not None:

        st.subheader("Top Languages by Repository Count")

        top = languages.nlargest(20, "repos").sort_values("repos")

        fig = px.bar(
            top,
            x="repos",
            y="primary_language_clean",
            orientation="h",
            title="Top 20 Programming Languages",
            labels={
                "repos": "Repositories",
                "primary_language_clean": "Language"
            }
        )
        show_chart(fig, 650)

        st.subheader("Language Popularity and Stars")

        fig = px.scatter(
            languages,
            x="repos",
            y="median_stars",
            size="repos",
            hover_name="primary_language_clean",
            title="Repository Count vs Median Stars",
            labels={
                "repos": "Repositories",
                "median_stars": "Median Stars"
            }
        )
        show_chart(fig)

        st.subheader("Language Activity")

        fig = px.bar(
            languages.nlargest(15, "repos"),
            x="primary_language_clean",
            y="share_active_365d",
            title="Recently Active Repository Share by Language",
            labels={
                "primary_language_clean": "Language",
                "share_active_365d": "Active within 365 days"
            }
        )
        fig.update_yaxes(tickformat=".0%")
        show_chart(fig)

    if language_bytes is not None:

        st.subheader("Language Code Size")

        top_bytes = language_bytes.nlargest(15, "total_bytes")

        fig = px.bar(
            top_bytes.sort_values("total_bytes"),
            x="total_bytes",
            y="language",
            orientation="h",
            title="Languages by Total Code Bytes",
            labels={
                "total_bytes": "Total Bytes",
                "language": "Language"
            }
        )
        show_chart(fig, 550)


# ============================================================
# 7. ACTIVITY AND COHORTS
# ============================================================

elif page == "Activity & Cohorts":

    st.header("📈 Repository Activity & Creation Cohorts")

    recency = get_data("recency_buckets.csv")
    cohorts = get_data("created_year_cohorts.csv")

    if recency is not None:

        recency = recency.sort_values("bucket_order")

        st.subheader("Activity by Last Push")

        fig = px.bar(
            recency,
            x="bucket",
            y="repos",
            title="Repository Recency Distribution",
            labels={
                "bucket": "Last Push Category",
                "repos": "Repositories"
            }
        )
        show_chart(fig)

        st.subheader("Median Stars by Activity Category")

        fig = px.bar(
            recency,
            x="bucket",
            y="median_stars",
            title="Median Stars by Last Push Category",
            labels={
                "bucket": "Last Push Category",
                "median_stars": "Median Stars"
            }
        )
        show_chart(fig)

        st.subheader("Pull Request Participation")

        fig = px.bar(
            recency,
            x="bucket",
            y="share_with_pull_requests",
            title="Share of Repositories with Pull Requests",
            labels={
                "bucket": "Last Push Category",
                "share_with_pull_requests": "Share"
            }
        )
        fig.update_yaxes(tickformat=".0%")
        show_chart(fig)

    if cohorts is not None:

        st.divider()
        st.subheader("Repository Creation Year Cohorts")

        cohorts = cohorts.sort_values("created_year")

        fig = px.line(
            cohorts,
            x="created_year",
            y="repos",
            markers=True,
            title="Repositories Created by Year",
            labels={
                "created_year": "Creation Year",
                "repos": "Repositories"
            }
        )
        show_chart(fig)

        fig = px.line(
            cohorts,
            x="created_year",
            y="median_stars",
            markers=True,
            title="Median Stars by Creation Year"
        )
        show_chart(fig)

        fig = px.line(
            cohorts,
            x="created_year",
            y="share_active_365d",
            markers=True,
            title="Recent Activity Share by Creation Cohort"
        )
        fig.update_yaxes(tickformat=".0%")
        show_chart(fig)


# ============================================================
# 8. REPOSITORY HEALTH
# ============================================================

elif page == "Repository Health":

    st.header("🛡️ Repository Health Analysis")

    archived = get_data("archived_vs_active.csv")
    license_flag = get_data("license_flag.csv")
    license_summary = get_data("license_summary.csv")

    if archived is not None:

        st.subheader("Archived vs Active Repositories")

        fig = px.bar(
            archived,
            x="isArchived",
            y="repos",
            color="isArchived",
            title="Archived and Active Repository Counts",
            labels={
                "isArchived": "Archived",
                "repos": "Repositories"
            }
        )
        show_chart(fig)

        st.subheader("Repository Characteristics")

        metrics = [
            "median_stars",
            "median_forks",
            "median_commits",
            "median_age_days",
            "median_days_since_push"
        ]

        selected_metric = st.selectbox(
            "Select a metric",
            metrics
        )

        fig = px.bar(
            archived,
            x="isArchived",
            y=selected_metric,
            color="isArchived",
            title=f"{selected_metric} by Archive Status"
        )
        show_chart(fig)

    if license_flag is not None:

        st.divider()
        st.subheader("Repositories with and without Licenses")

        fig = px.bar(
            license_flag,
            x="has_license",
            y="repos",
            color="has_license",
            title="License Availability",
            labels={
                "has_license": "Has License",
                "repos": "Repositories"
            }
        )
        show_chart(fig)

        fig = px.bar(
            license_flag,
            x="has_license",
            y="median_stars",
            title="Median Stars by License Availability"
        )
        show_chart(fig)

    if license_summary is not None:

        st.subheader("License Distribution")

        fig = px.bar(
            license_summary.nlargest(15, "repos").sort_values("repos"),
            x="repos",
            y="license_clean",
            orientation="h",
            title="Top Licenses by Repository Count"
        )
        show_chart(fig, 550)


# ============================================================
# 9. COLLABORATION ANALYSIS
# ============================================================

elif page == "Collaboration Analysis":

    st.header("🤝 Collaboration Analysis")

    collab = get_data("collab_buckets.csv")
    sizes = get_data("size_buckets.csv")

    if collab is not None:

        collab = collab.sort_values("bucket_order")

        st.subheader("Repositories by Collaboration Level")

        fig = px.bar(
            collab,
            x="bucket",
            y="repos",
            title="Repository Collaboration Buckets",
            labels={
                "bucket": "Collaboration Bucket",
                "repos": "Repositories"
            }
        )
        show_chart(fig)

        st.subheader("Median Stars by Collaboration Level")

        fig = px.bar(
            collab,
            x="bucket",
            y="median_stars",
            title="Median Stars by Collaboration Bucket"
        )
        show_chart(fig)

        st.subheader("Pull Request Participation")

        fig = px.bar(
            collab,
            x="bucket",
            y="share_with_pull_requests",
            title="Pull Request Share by Collaboration Bucket"
        )
        fig.update_yaxes(tickformat=".0%")
        show_chart(fig)

    if sizes is not None:

        st.divider()
        st.subheader("Repository Size Analysis")

        sizes = sizes.sort_values("bucket_order")

        fig = px.bar(
            sizes,
            x="bucket",
            y="repos",
            title="Repository Distribution by Size"
        )
        show_chart(fig)

        fig = px.bar(
            sizes,
            x="bucket",
            y="median_stars",
            title="Median Stars by Repository Size"
        )
        show_chart(fig)


# ============================================================
# 10. TOPICS ANALYSIS
# ============================================================

elif page == "Topics Analysis":

    st.header("🏷️ GitHub Topics Analysis")

    topics = get_data("topic_summary.csv")

    if topics is not None:

        st.subheader("Most Common Repository Topics")

        top_topics = topics.nlargest(20, "repos").sort_values("repos")

        fig = px.bar(
            top_topics,
            x="repos",
            y="topic",
            orientation="h",
            title="Top 20 GitHub Topics",
            labels={
                "repos": "Repositories",
                "topic": "Topic"
            }
        )
        show_chart(fig, 650)

        st.subheader("Topics and Popularity")

        fig = px.scatter(
            topics.nlargest(100, "repos"),
            x="repos",
            y="median_stars",
            size="repos",
            hover_name="topic",
            title="Topic Frequency vs Median Stars"
        )
        show_chart(fig)

        st.subheader("Topic Data")

        st.dataframe(
            topics.sort_values("repos", ascending=False),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

elif page == "Correlation Analysis":

    st.header("🔗 Repository Metric Correlations")

    corr = get_data("correlation_log1p.csv")

    if corr is not None:

        matrix = corr.set_index("variable")

        st.subheader("Correlation Heatmap")

        fig = px.imshow(
            matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Correlation Matrix of Log-Transformed Variables"
        )

        show_chart(fig, 700)

        st.subheader("Correlation with Stars")

        star_corr = corr[
            ["variable", "stars"]
        ].sort_values("stars", ascending=False)

        fig = px.bar(
            star_corr,
            x="stars",
            y="variable",
            orientation="h",
            title="Correlation of Variables with Stars",
            labels={
                "stars": "Correlation",
                "variable": "Variable"
            }
        )
        show_chart(fig, 550)

        st.info(
            "These are correlations between log-transformed metrics. "
            "Correlation does not establish causation."
        )


# ============================================================
# 12. MACHINE LEARNING RESULTS
# ============================================================

elif page == "Machine Learning Results":

    st.header("🤖 Machine Learning Analysis")

    results = get_data("ml_results.csv")
    importance = get_data("ml_feature_importance_gbt.csv")

    if results is None or results.empty:
        st.error("Machine learning results are unavailable.")
        st.stop()

    st.markdown("""
    **Classification objective:** Identify repositories with at least
    100 stars.
    """)

    feature_sets = results["feature_set"].dropna().unique().tolist()

    selected_set = st.selectbox(
        "Select feature set",
        feature_sets
    )

    filtered = results[
        results["feature_set"] == selected_set
    ].copy()

    st.subheader("Model Performance")

    metric_options = [
        "precision",
        "recall",
        "f1",
        "accuracy",
        "pr_auc",
        "roc_auc"
    ]

    selected_metric = st.selectbox(
        "Choose evaluation metric",
        metric_options,
        index=4
    )

    fig = px.bar(
        filtered,
        x="model",
        y=selected_metric,
        color="model",
        title=f"Model Comparison: {selected_metric.upper()}",
        labels={
            selected_metric: selected_metric.upper(),
            "model": "Model"
        }
    )
    fig.update_yaxes(tickformat=".0%")
    show_chart(fig)

    st.subheader("All Evaluation Metrics")

    st.dataframe(
        filtered[
            [
                "model",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "pr_auc",
                "roc_auc",
                "fit_seconds"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # Confusion matrix
    st.subheader("Confusion Matrix")

    selected_model = st.selectbox(
        "Select model for confusion matrix",
        filtered["model"].tolist()
    )

    model_row = filtered[
        filtered["model"] == selected_model
    ].iloc[0]

    confusion = pd.DataFrame(
        [
            [model_row["TN"], model_row["FP"]],
            [model_row["FN"], model_row["TP"]]
        ],
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"]
    )

    fig = px.imshow(
        confusion,
        text_auto=",",
        color_continuous_scale="Blues",
        title=f"Confusion Matrix: {selected_model}"
    )
    show_chart(fig, 380)

    # Feature importance
    if importance is not None:

        st.divider()
        st.subheader("Gradient-Boosted Trees Feature Importance")

        importance_sets = importance[
            "feature_set"
        ].dropna().unique().tolist()

        chosen_importance_set = st.selectbox(
            "Select feature set for feature importance",
            importance_sets
        )

        selected_importance = importance[
            importance["feature_set"] == chosen_importance_set
        ].nlargest(20, "importance").sort_values("importance")

        fig = px.bar(
            selected_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 20 Feature Importances"
        )
        show_chart(fig, 600)

    st.warning(
        "Important limitation: forks and watchers are themselves "
        "signals of repository popularity. Models that include them "
        "have a partly circular advantage. Compare the feature sets "
        "carefully; these results do not demonstrate future prediction."
    )


# ============================================================
# 13. DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.header("🔎 Analytical Data Explorer")

    if not csv_files:
        st.error("No CSV files found in the data folder.")
        st.stop()

    selected_file = st.selectbox(
        "Choose a dataset",
        [file.name for file in csv_files]
    )

    df = load_csv(selected_file)

    if df is not None:

        st.subheader(selected_file)

        c1, c2, c3 = st.columns(3)

        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        with st.expander("Column Information"):

            info = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isna().sum().values
            })

            st.dataframe(
                info,
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False),
            file_name=selected_file,
            mime="text/csv"
        )


# ============================================================
# 14. FOOTER
# ============================================================

st.sidebar.divider()
st.sidebar.caption("GitHub Repository Analytics Project")
st.sidebar.caption("Built with Python | Streamlit | Plotly")
