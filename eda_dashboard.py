"""
Single-File Interactive EDA Dashboard
Created for classroom demonstration of Exploratory Data Analysis (EDA)
using Python, Pandas, NumPy, Matplotlib, Seaborn, and Flask.

File: eda_dashboard.py
Dataset: StudentsPerformance.csv
"""

import os
import io
import base64
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template_string

CSV_PATH = "StudentsPerformance.csv"

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = False

def fig_to_base64(fig):
    """Utility to convert a Matplotlib figure into a base64 encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"

# ==============================================================================
# 1. LOAD DATASET
# ==============================================================================
def load_dataset(filepath=CSV_PATH):
    """
    Loads the dataset from the specified CSV filepath.
    Returns a pandas DataFrame, or None if the file is not found or fails to load.
    """
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

# ==============================================================================
# 2. CLEAN DATASET
# ==============================================================================
def clean_dataset(df):
    """
    Performs data cleaning:
    - Removes whitespace from string columns
    - Verifies numerical types for score columns
    - Handles missing values if any
    """
    if df is None:
        return None
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    score_cols = [c for c in df_clean.columns if 'score' in c.lower()]
    for col in score_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean

# ==============================================================================
# 3. GET DATASET SUMMARY
# ==============================================================================
def get_dataset_summary(df):
    """
    Computes comprehensive dataset summary statistics for the dashboard overview.
    """
    if df is None:
        return {
            "status": "error",
            "message": f"CSV file '{CSV_PATH}' was not found. Please ensure it is in the same directory."
        }
    
    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)
    
    math_avg = round(float(df['math score'].mean()), 2) if 'math score' in df.columns else 0.0
    reading_avg = round(float(df['reading score'].mean()), 2) if 'reading score' in df.columns else 0.0
    writing_avg = round(float(df['writing score'].mean()), 2) if 'writing score' in df.columns else 0.0
    
    missing_dict = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    
    preview_df = df.head(5).copy()
    preview_records = preview_df.to_dict(orient='records')
    
    stats_df = df.describe().round(2).reset_index()
    stats_records = stats_df.to_dict(orient='records')
    
    return {
        "status": "success",
        "total_students": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "avg_math_score": math_avg,
        "avg_reading_score": reading_avg,
        "avg_writing_score": writing_avg,
        "total_missing": total_missing,
        "missing_per_column": missing_dict,
        "duplicate_count": duplicate_count,
        "preview_data": preview_records,
        "summary_stats": stats_records
    }

# ==============================================================================
# 4. GET NUMERIC COLUMNS
# ==============================================================================
def get_numeric_columns(df):
    """Returns a list of all numeric column names in the dataset."""
    if df is None:
        return []
    return df.select_dtypes(include=[np.number]).columns.tolist()

# ==============================================================================
# 5. GET CATEGORICAL COLUMNS
# ==============================================================================
def get_categorical_columns(df):
    """Returns a list of all categorical column names in the dataset."""
    if df is None:
        return []
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

# ==============================================================================
# 6. CREATE HISTOGRAM
# ==============================================================================
def create_histogram(df, column):
    """Creates a histogram with a KDE overlay for a continuous numeric variable."""
    if column not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.histplot(data=df, x=column, kde=True, bins=15, color='#4338CA', edgecolor='white', ax=ax)
    mean_val = df[column].mean()
    median_val = df[column].median()
    ax.axvline(mean_val, color='#DC2626', linestyle='--', linewidth=2, label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color='#16A34A', linestyle=':', linewidth=2, label=f"Median: {median_val:.1f}")
    ax.set_title(f"Histogram & KDE Distribution: {column.title()}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(column.title(), fontsize=11)
    ax.set_ylabel("Student Count (Frequency)", fontsize=11)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    return fig_to_base64(fig)

# ==============================================================================
# 7. CREATE BAR CHART
# ==============================================================================
def create_bar_chart(df, x_column, y_column):
    """Creates a bar chart comparing average values across categories."""
    if x_column not in df.columns or y_column not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    grouped = df.groupby(x_column)[y_column].mean().sort_values(ascending=False).reset_index()
    palette = sns.color_palette("crest", n_colors=len(grouped))
    bars = sns.barplot(data=grouped, x=x_column, y=y_column, hue=x_column, palette=palette, legend=False, ax=ax)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.1f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 4),
                        textcoords='offset points')
    ax.set_title(f"Average {y_column.title()} by {x_column.title()}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(x_column.title(), fontsize=11)
    ax.set_ylabel(f"Mean {y_column.title()}", fontsize=11)
    if len(str(grouped[x_column].iloc[0])) > 6 or len(grouped) > 4:
        plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    return fig_to_base64(fig)

# ==============================================================================
# 8. CREATE LINE CHART
# ==============================================================================
def create_line_chart(df, x_column, y_column):
    """Creates a line chart for meaningful ordered comparisons across categories."""
    if x_column not in df.columns or y_column not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    
    edu_order = [
        "some high school", "high school", "some college", 
        "associate's degree", "bachelor's degree", "master's degree"
    ]
    if x_column == "parental level of education":
        grouped = df.groupby(x_column)[y_column].mean().reindex([e for e in edu_order if e in df[x_column].unique()]).reset_index()
    else:
        grouped = df.groupby(x_column)[y_column].mean().sort_values().reset_index()
    
    ax.plot(grouped[x_column], grouped[y_column], marker='o', markersize=8, linewidth=2.5, color='#0284C7', label=f"Mean {y_column}")
    for idx, row in grouped.iterrows():
        ax.annotate(f"{row[y_column]:.1f}", (row[x_column], row[y_column]),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 7),
                    textcoords='offset points')
    ax.set_title(f"Trend / Progression of {y_column.title()} across {x_column.title()}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(x_column.title(), fontsize=11)
    ax.set_ylabel(f"Mean {y_column.title()}", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    return fig_to_base64(fig)

# ==============================================================================
# 9. CREATE BOX PLOT
# ==============================================================================
def create_box_plot(df, column, by_column=None):
    """Creates a univariate or bivariate box plot to show quartiles, median, and outliers."""
    if column not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    if by_column and by_column in df.columns:
        palette = sns.color_palette("Set2", n_colors=df[by_column].nunique())
        sns.boxplot(data=df, x=by_column, y=column, hue=by_column, palette=palette, legend=False, ax=ax, width=0.5, boxprops=dict(alpha=0.85))
        ax.set_title(f"Distribution of {column.title()} by {by_column.title()}", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(by_column.title(), fontsize=11)
        if df[by_column].nunique() > 3:
            plt.xticks(rotation=25, ha='right')
    else:
        sns.boxplot(data=df, y=column, color='#A78BFA', ax=ax, width=0.35, boxprops=dict(alpha=0.85))
        ax.set_title(f"Box Plot: Five-Number Summary of {column.title()}", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel(column.title(), fontsize=11)
    plt.tight_layout()
    return fig_to_base64(fig)

# ==============================================================================
# 10. CREATE SCATTER PLOT
# ==============================================================================
def create_scatter_plot(df, x_column, y_column, hue_column=None):
    """Creates a scatter plot to observe correlation and clustering between two variables."""
    if x_column not in df.columns or y_column not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    is_x_num = pd.api.types.is_numeric_dtype(df[x_column])
    is_y_num = pd.api.types.is_numeric_dtype(df[y_column])

    if is_x_num and is_y_num:
        if hue_column and hue_column in df.columns:
            sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column, palette='tab10', alpha=0.75, s=60, ax=ax)
            ax.legend(title=hue_column.title(), frameon=True, facecolor='white', framealpha=0.9)
        else:
            sns.scatterplot(data=df, x=x_column, y=y_column, color='#2563EB', alpha=0.65, s=60, ax=ax, label='Data Points')
            sns.regplot(data=df, x=x_column, y=y_column, scatter=False, ax=ax, color='#DC2626',
                        label='Trend Line', line_kws={'linestyle': '--', 'linewidth': 2})
            ax.legend(frameon=True, facecolor='white')
        corr_val = df[x_column].corr(df[y_column])
        ax.set_title(f"{x_column.title()} vs. {y_column.title()} (Pearson r = {corr_val:.2f})", fontsize=13, fontweight='bold', pad=12)
    else:
        sns.stripplot(data=df, x=x_column, y=y_column, hue=hue_column if hue_column in df.columns else None,
                      jitter=0.25, alpha=0.6, size=6, palette='tab10' if hue_column else None, ax=ax)
        ax.set_title(f"Scatter / Strip Distribution: {y_column.title()} by {x_column.title()}", fontsize=13, fontweight='bold', pad=12)
        if not is_x_num and df[x_column].nunique() > 3:
            plt.xticks(rotation=25, ha='right')

    ax.set_xlabel(x_column.title(), fontsize=11)
    ax.set_ylabel(y_column.title(), fontsize=11)
    plt.tight_layout()
    return fig_to_base64(fig)

# ==============================================================================
# 11. CREATE HEATMAP
# ==============================================================================
def create_heatmap(df):
    """Creates a correlation heatmap showing pairwise relationships between numeric variables."""
    num_cols = get_numeric_columns(df)
    if len(num_cols) < 2:
        return None
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    clean_labels = [c.title() for c in num_cols]
    sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', vmin=-1, vmax=1,
                xticklabels=clean_labels, yticklabels=clean_labels,
                cbar_kws={'label': 'Pearson Correlation (r)'},
                linewidths=1, linecolor='white', ax=ax, square=True)
    ax.set_title("Correlation Heatmap: Exam Scores", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig_to_base64(fig)

# ==============================================================================
# 12. CREATE COUNT PLOT
# ==============================================================================
def create_count_plot(df, column):
    """Creates a count plot showing frequencies of categories."""
    if column not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    order = df[column].value_counts().index
    palette = sns.color_palette("mako", n_colors=len(order))
    sns.countplot(data=df, x=column, order=order, hue=column, palette=palette, legend=False, ax=ax)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 4),
                        textcoords='offset points')
    ax.set_title(f"Frequency Distribution of {column.title()}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(column.title(), fontsize=11)
    ax.set_ylabel("Student Count", fontsize=11)
    if len(str(order[0])) > 6 or len(order) > 4:
        plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    return fig_to_base64(fig)

# ==============================================================================
# 13. CREATE PIE CHART
# ==============================================================================
def create_pie_chart(df, column):
    """Creates a pie chart showing proportions of a categorical column."""
    if column not in df.columns:
        return None
    counts = df[column].value_counts()
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    colors = sns.color_palette("pastel", n_colors=len(counts))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for text in texts:
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    ax.set_title(f"Proportional Share: {column.title()}", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig_to_base64(fig)

# ==============================================================================
# 14. CREATE PAIRPLOT
# ==============================================================================
def create_pairplot(df):
    """Creates a Seaborn pairplot of all numerical score variables."""
    num_cols = get_numeric_columns(df)
    cols_to_use = num_cols.copy()
    hue = 'gender' if 'gender' in df.columns else None
    if hue:
        cols_to_use.append(hue)
    
    palette = {'female': '#EC4899', 'male': '#3B82F6'} if hue == 'gender' else 'tab10'
    g = sns.pairplot(df[cols_to_use], hue=hue, palette=palette, corner=False, diag_kind='kde')
    g.fig.suptitle("Pairwise Bivariate Relationships & Distributions", y=1.02, fontsize=14, fontweight='bold')
    buf = io.BytesIO()
    g.savefig(buf, format='png', bbox_inches='tight', dpi=95)
    plt.close(g.fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"

# ==============================================================================
# 15. GET CHART EXPLANATION
# ==============================================================================
def get_chart_explanation(chart_type, x_column=None, y_column=None):
    """Returns educational explanations, usage criteria, and observations for graphs."""
    explanations = {
        "histogram": {
            "title": "Histogram & KDE Curve (Univariate Analysis)",
            "purpose": "Visualizes the frequency distribution, center, spread, and skewness of a single continuous numerical variable.",
            "when_to_use": "Use when inspecting whether continuous data is normally distributed, skewed to the left/right, multimodal, or bounded.",
            "what_to_observe": "In this students dataset, Math, Reading, and Writing scores all show unimodal, approximately bell-shaped (normal) distributions centered around 66–69 points.",
            "data_type": "1 Continuous Numerical variable",
            "limitations": "Bin width choices can distort appearance. Does not reveal individual data point values."
        },
        "bar_chart": {
            "title": "Bar Chart (Bivariate Aggregation)",
            "purpose": "Compares an aggregated summary metric (such as mean or sum) across discrete categorical groups.",
            "when_to_use": "Use when comparing average performance, totals, or percentages across demographic categories.",
            "what_to_observe": "Students with standard lunch consistently score higher across all subjects compared to students with free/reduced lunch. Higher parental education levels also correlate with higher test score averages.",
            "data_type": "1 Categorical variable (X) + 1 Numerical variable (Y)",
            "limitations": "Hides the internal spread, standard deviation, and outliers within each category (use a Box Plot for distribution comparisons)."
        },
        "line_chart": {
            "title": "Line Chart (Trend / Ordered Progression)",
            "purpose": "Displays a progression, trend, or trajectory across an ordered sequence or continuous dimension.",
            "when_to_use": "Use when the X-axis features inherently ordered levels (such as parental education tiers, grade levels, or time series).",
            "what_to_observe": "Following parental education from 'some high school' through 'master's degree' shows a steady upward slope in student exam averages.",
            "data_type": "1 Ordered Categorical or Time variable + 1 Continuous Numerical variable",
            "limitations": "Misleading if categories along the X-axis have no natural sequential or ordinal relationship."
        },
        "box_plot": {
            "title": "Box Plot (Five-Number Summary & Outlier Detection)",
            "purpose": "Displays the median, quartiles (Q1, Q3), interquartile range (IQR), whiskers, and outliers.",
            "when_to_use": "Ideal for comparing medians and data variance across different categories, and spotting low or high outliers.",
            "what_to_observe": "Female students have higher median scores and tighter IQRs in Reading and Writing. A few severe low-score outliers exist below 25 points in Math.",
            "data_type": "1 Numerical variable (univariate) or 1 Categorical + 1 Numerical (bivariate)",
            "limitations": "Does not convey sample size in each category or show whether a distribution is bimodal."
        },
        "scatter_plot": {
            "title": "Scatter Plot (Bivariate Correlation)",
            "purpose": "Examines the relationship, direction, and strength of association between two continuous numerical variables.",
            "when_to_use": "Use when checking whether changes in one numeric measurement predict or relate to changes in another.",
            "what_to_observe": "Reading and Writing scores exhibit an exceptionally strong positive linear correlation (r ≈ 0.95). Students who excel in reading almost universally score high in writing.",
            "data_type": "2 Continuous Numerical variables (with optional Categorical Hue)",
            "limitations": "Points can overlap heavily (overplotting) in large datasets. Correlation does not imply causation."
        },
        "heatmap": {
            "title": "Correlation Heatmap (Multivariate Correlation Matrix)",
            "purpose": "Uses a 2D color matrix to instantly communicate pairwise Pearson correlation coefficients (r).",
            "when_to_use": "Use during initial multivariate EDA to scan all numerical features for collinearity and relationships.",
            "what_to_observe": "All three academic disciplines are strongly positively correlated (r > 0.80), indicating high overall academic consistency among students.",
            "data_type": "Matrix of Numerical variables",
            "limitations": "Only captures linear relationships; non-linear relationships will appear with low or misleading coefficients."
        },
        "count_plot": {
            "title": "Count Plot (Categorical Frequency Distribution)",
            "purpose": "Visualizes the exact number of occurrences for each discrete category.",
            "when_to_use": "Use to check class balance or imbalance across demographics.",
            "what_to_observe": "Gender is evenly balanced (~52% female, 48% male). Race/Ethnicity has the highest representation in Groups C and D, while Group A has the fewest students.",
            "data_type": "1 Categorical variable",
            "limitations": "Only reports frequencies; cannot show relationships with student exam performance directly."
        },
        "pie_chart": {
            "title": "Pie Chart (Proportional Composition)",
            "purpose": "Shows the fractional breakdown of parts relative to a 100% whole.",
            "when_to_use": "Use only when comparing 2 to 5 distinct categories where showing percentage of total is the primary objective.",
            "what_to_observe": "Approximately 64.5% of students have standard lunch, while 35.5% receive subsidized/free lunch. ~64% did not take a test preparation course.",
            "data_type": "1 Categorical variable with few categories",
            "limitations": "Human perception struggles to accurately distinguish subtle differences in angles and slice areas. Bar charts are generally preferred."
        },
        "pairplot": {
            "title": "Pair Plot (Multivariate Pairwise Grid)",
            "purpose": "Plots all bivariate scatter plots across numeric columns simultaneously, with univariate distributions along the diagonal.",
            "when_to_use": "Use at the start of EDA to scan all multi-variable interactions in a single visual pass.",
            "what_to_observe": "Shows strong positive collinearity across all subjects. Adding a gender hue highlights that females dominate the upper clusters in Reading/Writing while males shift slightly higher in Math.",
            "data_type": "3+ Continuous Numerical variables + Optional Categorical Hue",
            "limitations": "Computationally demanding on massive datasets; can quickly become crowded if feature count is high."
        }
    }
    key = chart_type.lower()
    return explanations.get(key, {
        "title": f"{chart_type.title()} Chart",
        "purpose": "Exploratory visualization of dataset relationships.",
        "when_to_use": "Use to explore patterns, outliers, and trends.",
        "what_to_observe": "Look for clusters, skewness, and group differences.",
        "data_type": "Numerical or Categorical",
        "limitations": "Requires careful interpretation."
    })

# ==============================================================================
# 16. GENERATE MATPLOTLIB CODE
# ==============================================================================
def generate_matplotlib_code(chart_type, x_column="math score", y_column="reading score", hue_column=None):
    """Generates beginner-friendly, clean, runnable Matplotlib code."""
    c_type = chart_type.lower()
    if c_type == "histogram":
        return f"""# Matplotlib: Histogram with Mean Line
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(df['{x_column}'], bins=15, color='#4338CA', edgecolor='white', alpha=0.8)
mean_val = df['{x_column}'].mean()
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {{mean_val:.1f}}')

plt.title('Distribution of {x_column.title()}', fontsize=14, fontweight='bold')
plt.xlabel('{x_column.title()}', fontsize=12)
plt.ylabel('Frequency (Student Count)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()"""

    elif c_type == "bar_chart":
        return f"""# Matplotlib: Bar Chart of Category Means
import matplotlib.pyplot as plt

# Step 1: Pre-calculate the mean values (Matplotlib requires explicit aggregation)
grouped = df.groupby('{x_column}')['{y_column}'].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
bars = plt.bar(grouped.index, grouped.values, color='#0D9488', edgecolor='black')

# Step 2: Annotate values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{{yval:.1f}}', ha='center', fontweight='bold')

plt.title('Mean {y_column.title()} by {x_column.title()}', fontsize=14, fontweight='bold')
plt.xlabel('{x_column.title()}', fontsize=12)
plt.ylabel('Average {y_column.title()}', fontsize=12)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()"""

    elif c_type == "line_chart":
        return f"""# Matplotlib: Ordered Trend Line Chart
import matplotlib.pyplot as plt

# Group and order the data
grouped = df.groupby('{x_column}')['{y_column}'].mean().sort_values()

plt.figure(figsize=(8, 5))
plt.plot(grouped.index, grouped.values, marker='o', linewidth=2.5, color='#0284C7')

plt.title('Trend of {y_column.title()} across {x_column.title()}', fontsize=14, fontweight='bold')
plt.xlabel('{x_column.title()}', fontsize=12)
plt.ylabel('Mean {y_column.title()}', fontsize=12)
plt.xticks(rotation=25, ha='right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()"""

    elif c_type == "box_plot":
        return f"""# Matplotlib: Box Plot
import matplotlib.pyplot as plt

# Matplotlib requires splitting data into separate lists for each group
categories = df['{x_column}'].unique()
data_per_group = [df[df['{x_column}'] == cat]['{y_column}'].dropna() for cat in categories]

plt.figure(figsize=(8, 5))
plt.boxplot(data_per_group, tick_labels=categories, patch_artist=True,
            boxprops=dict(facecolor='#A78BFA', color='black'))

plt.title('{y_column.title()} by {x_column.title()}', fontsize=14, fontweight='bold')
plt.xlabel('{x_column.title()}', fontsize=12)
plt.ylabel('{y_column.title()}', fontsize=12)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()"""

    elif c_type == "scatter_plot":
        return f"""# Matplotlib: Scatter Plot with Manual Trend Line
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 5))
plt.scatter(df['{x_column}'], df['{y_column}'], color='#2563EB', alpha=0.65, edgecolors='none')

# Calculate trend line using numpy polyfit
m, b = np.polyfit(df['{x_column}'], df['{y_column}'], 1)
plt.plot(df['{x_column}'], m*df['{x_column}'] + b, color='red', linestyle='--', linewidth=2, label='Trend Line')

plt.title('{x_column.title()} vs. {y_column.title()}', fontsize=14, fontweight='bold')
plt.xlabel('{x_column.title()}', fontsize=12)
plt.ylabel('{y_column.title()}', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()"""

    elif c_type == "heatmap":
        return """# Matplotlib: Heatmap (Requires imshow + nested annotation loops)
import matplotlib.pyplot as plt
import numpy as np

numeric_df = df[['math score', 'reading score', 'writing score']]
corr = numeric_df.corr().values
labels = ['Math', 'Reading', 'Writing']

fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.imshow(corr, cmap='Blues', vmin=-1, vmax=1)
fig.colorbar(cax)

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# Matplotlib requires manual nested loops to print text inside cells
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', color='black', fontweight='bold')

plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""

    elif c_type == "count_plot":
        return f"""# Matplotlib: Count Plot (Requires value_counts first)
import matplotlib.pyplot as plt

counts = df['{x_column}'].value_counts()

plt.figure(figsize=(8, 5))
bars = plt.bar(counts.index, counts.values, color='#0D9488')

for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{{int(bar.get_height())}}', ha='center')

plt.title('Frequency of {x_column.title()}', fontsize=14, fontweight='bold')
plt.xlabel('{x_column.title()}', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()"""

    elif c_type == "pie_chart":
        return f"""# Matplotlib: Pie Chart
import matplotlib.pyplot as plt

counts = df['{x_column}'].value_counts()

plt.figure(figsize=(7, 6))
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140,
        colors=['#60A5FA', '#F472B6', '#34D399', '#FBBF24', '#A78BFA'])

plt.title('Proportions of {x_column.title()}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""

    elif c_type == "pairplot":
        return """# Matplotlib: Pair Plot (Must manually create subplots in a grid)
import matplotlib.pyplot as plt

cols = ['math score', 'reading score', 'writing score']
fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for i, col1 in enumerate(cols):
    for j, col2 in enumerate(cols):
        ax = axes[i, j]
        if i == j:
            ax.hist(df[col1], bins=15, color='#4338CA', edgecolor='white')
        else:
            ax.scatter(df[col2], df[col1], alpha=0.5, s=20, color='#3B82F6')
        if i == 2: ax.set_xlabel(col2)
        if j == 0: ax.set_ylabel(col1)

plt.suptitle('Pairwise Grid via Matplotlib', y=1.01, fontsize=14)
plt.tight_layout()
plt.show()"""

    return "# Code not available for this chart type."

# ==============================================================================
# 17. GENERATE SEABORN CODE
# ==============================================================================
def generate_seaborn_code(chart_type, x_column="math score", y_column="reading score", hue_column=None):
    """Generates beginner-friendly, clean, runnable Seaborn code."""
    c_type = chart_type.lower()
    if c_type == "histogram":
        return f"""# Seaborn: Histogram with automatic KDE curve
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='{x_column}', kde=True, bins=15, color='#4338CA')

plt.title('Distribution of {x_column.title()}', fontsize=14, fontweight='bold')
plt.xlabel('{x_column.title()}', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()"""

    elif c_type == "bar_chart":
        return f"""# Seaborn: Bar Chart with automatic Mean calculation
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
# Seaborn calculates means and confidence intervals directly from the DataFrame!
sns.barplot(data=df, x='{x_column}', y='{y_column}', palette='crest', errorbar=None)

plt.title('Mean {y_column.title()} by {x_column.title()}', fontsize=14, fontweight='bold')
plt.xlabel('{x_column.title()}', fontsize=12)
plt.ylabel('Average {y_column.title()}', fontsize=12)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()"""

    elif c_type == "line_chart":
        return f"""# Seaborn: Line Plot with automatic category aggregation
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='{x_column}', y='{y_column}', marker='o', linewidth=2.5, color='#0284C7', errorbar=None)

plt.title('Trend of {y_column.title()} across {x_column.title()}', fontsize=14, fontweight='bold')
plt.xlabel('{x_column.title()}', fontsize=12)
plt.ylabel('Mean {y_column.title()}', fontsize=12)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()"""

    elif c_type == "box_plot":
        return f"""# Seaborn: Grouped Box Plot in one line
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
# Pass column names directly — no manual grouping needed!
sns.boxplot(data=df, x='{x_column}', y='{y_column}', palette='Set2')

plt.title('{y_column.title()} by {x_column.title()}', fontsize=14, fontweight='bold')
plt.xlabel('{x_column.title()}', fontsize=12)
plt.ylabel('{y_column.title()}', fontsize=12)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()"""

    elif c_type == "scatter_plot":
        hue_param = f", hue='{hue_column}'" if hue_column else ""
        return f"""# Seaborn: Scatter Plot with automatic legend & hue
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
# One parameter 'hue' handles all grouping and legend generation
sns.scatterplot(data=df, x='{x_column}', y='{y_column}'{hue_param}, palette='tab10', alpha=0.7, s=60)

plt.title('{x_column.title()} vs. {y_column.title()}', fontsize=14, fontweight='bold')
plt.xlabel('{x_column.title()}', fontsize=12)
plt.ylabel('{y_column.title()}', fontsize=12)
plt.tight_layout()
plt.show()"""

    elif c_type == "heatmap":
        return """# Seaborn: Annotated Correlation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

numeric_df = df[['math score', 'reading score', 'writing score']]
corr = numeric_df.corr()

plt.figure(figsize=(7, 5))
# annot=True handles all cell text annotations automatically!
sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', vmin=-1, vmax=1, square=True)

plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""

    elif c_type == "count_plot":
        return f"""# Seaborn: Count Plot for Categorical Data
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
# Seaborn computes value frequencies automatically!
sns.countplot(data=df, x='{x_column}', palette='mako')

plt.title('Frequency Distribution of {x_column.title()}', fontsize=14, fontweight='bold')
plt.xlabel('{x_column.title()}', fontsize=12)
plt.ylabel('Student Count', fontsize=12)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()"""

    elif c_type == "pie_chart":
        return f"""# Note: Seaborn intentionally does NOT have a pie chart function.
# The Seaborn library author recommends bar or count plots for clearer perception!
# Use Matplotlib for pie charts, or Seaborn countplot instead:
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x='{x_column}', palette='pastel')
plt.title('Frequency of {x_column.title()} (Recommended over Pie Charts)')
plt.show()"""

    elif c_type == "pairplot":
        return """# Seaborn: Pair Plot in a single command
import seaborn as sns
import matplotlib.pyplot as plt

# Generates entire 3x3 matrix of bivariate scatter plots and univariate KDEs!
sns.pairplot(df[['math score', 'reading score', 'writing score', 'gender']], hue='gender', palette='Set1')
plt.show()"""

    return "# Code not available for this chart type."

# ==============================================================================
# 18. CREATE DASHBOARD HTML
# ==============================================================================
def create_dashboard_html():
    """Generates the full-featured, single-file HTML, CSS, and Vanilla JavaScript."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EDA Classroom Dashboard: Students Performance in Exams</title>
  <style>
    :root {
      --primary: #4338CA;
      --primary-light: #EEF2FF;
      --primary-dark: #312E81;
      --secondary: #0D9488;
      --accent: #F59E0B;
      --danger: #EF4444;
      --success: #10B981;
      --bg: #F8FAFC;
      --card-bg: #FFFFFF;
      --text: #1E293B;
      --text-muted: #64748B;
      --border: #E2E8F0;
      --code-bg: #0F172A;
      --code-text: #E2E8F0;
      --radius: 10px;
      --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding-bottom: 40px;
    }

    /* Header */
    header {
      background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
      color: white;
      padding: 24px 32px;
      box-shadow: var(--shadow-lg);
    }
    .header-content {
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    .header-title h1 {
      font-size: 1.75rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .header-title p {
      font-size: 0.95rem;
      color: #C7D2FE;
      margin-top: 4px;
    }
    .badge-dataset {
      background: rgba(255, 255, 255, 0.15);
      border: 1px solid rgba(255, 255, 255, 0.25);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Container */
    .container {
      max-width: 1200px;
      margin: 24px auto 0;
      padding: 0 20px;
    }

    /* Alerts */
    .alert-error {
      background-color: #FEF2F2;
      border: 1px solid #FCA5A5;
      color: #991B1B;
      padding: 16px;
      border-radius: var(--radius);
      margin-bottom: 20px;
      font-size: 0.95rem;
    }

    /* Tabs Navigation */
    .tab-bar {
      display: flex;
      background: white;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 6px;
      gap: 6px;
      overflow-x: auto;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }
    .tab-btn {
      flex: 1;
      min-width: 150px;
      padding: 10px 16px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-weight: 600;
      font-size: 0.9rem;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.2s ease;
      white-space: nowrap;
    }
    .tab-btn:hover {
      background: var(--bg);
      color: var(--primary);
    }
    .tab-btn.active {
      background: var(--primary);
      color: white;
      box-shadow: 0 2px 4px rgba(67, 56, 202, 0.25);
    }

    /* Tab Content Panels */
    .tab-panel {
      display: none;
      animation: fadeIn 0.25s ease-in-out;
    }
    .tab-panel.active {
      display: block;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Cards */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }
    .card-header h2 {
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--primary-dark);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Metrics Grid */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .metric-card {
      background: white;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px 20px;
      box-shadow: var(--shadow);
      border-left: 4px solid var(--primary);
    }
    .metric-card.teal { border-left-color: var(--secondary); }
    .metric-card.amber { border-left-color: var(--accent); }
    .metric-card.emerald { border-left-color: var(--success); }
    .metric-label {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      font-weight: 600;
    }
    .metric-value {
      font-size: 1.8rem;
      font-weight: 700;
      color: var(--text);
      margin-top: 4px;
    }
    .metric-sub {
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 2px;
    }

    /* Tables */
    .table-container {
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-top: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.88rem;
    }
    th {
      background-color: #F1F5F9;
      color: #334155;
      font-weight: 600;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }
    td {
      padding: 10px 14px;
      border-bottom: 1px solid #F1F5F9;
      color: #475569;
      white-space: nowrap;
    }
    tr:last-child td {
      border-bottom: none;
    }
    tr:hover {
      background-color: #F8FAFC;
    }

    /* Form Controls & Toolbar */
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
      background: #F8FAFC;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      margin-bottom: 20px;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .form-group label {
      font-size: 0.82rem;
      font-weight: 600;
      color: #475569;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }
    select, button.btn {
      padding: 8px 14px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.9rem;
      background: white;
      color: var(--text);
      cursor: pointer;
      outline: none;
    }
    select:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 2px rgba(67, 56, 202, 0.15);
    }
    .btn-primary {
      background-color: var(--primary);
      color: white;
      border: none;
      font-weight: 600;
      transition: background 0.2s;
    }
    .btn-primary:hover {
      background-color: var(--primary-dark);
    }

    /* Chart Layout */
    .chart-layout {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 20px;
    }
    @media (max-width: 900px) {
      .chart-layout {
        grid-template-columns: 1fr;
      }
    }
    .chart-box {
      background: white;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 420px;
      box-shadow: var(--shadow);
      position: relative;
    }
    .chart-img {
      max-width: 100%;
      height: auto;
      border-radius: 6px;
      display: block;
    }
    .spinner {
      border: 4px solid #F1F5F9;
      border-top: 4px solid var(--primary);
      border-radius: 50%;
      width: 36px;
      height: 36px;
      animation: spin 0.8s linear infinite;
      margin-bottom: 12px;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    /* Explanation & Code Cards */
    .info-card {
      background: white;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .info-item h4 {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--primary);
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .info-item p {
      font-size: 0.92rem;
      color: #334155;
    }

    /* Code Block Section */
    .code-container {
      margin-top: 16px;
      border: 1px solid #334155;
      border-radius: 8px;
      overflow: hidden;
      background: var(--code-bg);
    }
    .code-header {
      background: #1E293B;
      padding: 8px 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .code-tabs {
      display: flex;
      gap: 8px;
    }
    .code-tab-btn {
      background: transparent;
      border: none;
      color: #94A3B8;
      font-size: 0.82rem;
      font-weight: 600;
      padding: 4px 10px;
      cursor: pointer;
      border-radius: 4px;
    }
    .code-tab-btn.active {
      background: #334155;
      color: #F8FAFC;
    }
    .copy-btn {
      background: #334155;
      color: #E2E8F0;
      border: none;
      font-size: 0.78rem;
      padding: 4px 8px;
      border-radius: 4px;
      cursor: pointer;
    }
    .copy-btn:hover {
      background: #475569;
    }
    pre.code-box {
      padding: 14px;
      color: var(--code-text);
      font-family: Consolas, Monaco, "Courier New", monospace;
      font-size: 0.82rem;
      overflow-x: auto;
      max-height: 260px;
      line-height: 1.45;
    }

    /* Side by side comparison */
    .comparison-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-top: 16px;
    }
    @media (max-width: 800px) {
      .comparison-grid {
        grid-template-columns: 1fr;
      }
    }
    .callout-box {
      background: #F0FDF4;
      border-left: 4px solid var(--success);
      padding: 16px;
      border-radius: 6px;
      margin: 16px 0;
      font-size: 0.92rem;
      color: #166534;
    }
    .callout-box.info {
      background: #EFF6FF;
      border-left-color: #3B82F6;
      color: #1E40AF;
    }
    .callout-box h4 {
      font-weight: 700;
      margin-bottom: 4px;
    }
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="header-content">
      <div class="header-title">
        <h1>🎓 EDA Classroom Dashboard</h1>
        <p>Interactive Data Science Teaching Tool: Exploratory Data Analysis with Matplotlib & Seaborn</p>
      </div>
      <div class="badge-dataset">
        <span>📁 Dataset:</span>
        <strong>StudentsPerformance.csv</strong>
        <span id="headerStudentCount" style="opacity: 0.85;">(1,000 Students)</span>
      </div>
    </div>
  </header>

  <div class="container">
    <!-- Error notification if CSV not found -->
    <div id="errorBanner" class="alert-error" style="display: none;"></div>

    <!-- Navigation Tabs -->
    <nav class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('overview')">📋 1. Overview & Primer</button>
      <button class="tab-btn" onclick="switchTab('univariate')">📊 2. Univariate Analysis</button>
      <button class="tab-btn" onclick="switchTab('bivariate')">🔀 3. Bivariate Analysis</button>
      <button class="tab-btn" onclick="switchTab('multivariate')">🌐 4. Multivariate Analysis</button>
      <button class="tab-btn" onclick="switchTab('comparison')">⚔️ 5. Matplotlib vs Seaborn</button>
      <button class="tab-btn" onclick="switchTab('guide')">📑 6. Graph Comparison Guide</button>
    </nav>

    <!-- ===================================================================== -->
    <!-- TAB 1: OVERVIEW & EDA PRIMER -->
    <!-- ===================================================================== -->
    <section id="tab-overview" class="tab-panel active">
      <!-- Summary Metrics Grid -->
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-label">Total Students</div>
          <div class="metric-value" id="mTotalStudents">-</div>
          <div class="metric-sub">Observations (Rows)</div>
        </div>
        <div class="metric-card teal">
          <div class="metric-label">Features / Columns</div>
          <div class="metric-value" id="mTotalColumns">-</div>
          <div class="metric-sub">3 Numeric, 5 Categorical</div>
        </div>
        <div class="metric-card amber">
          <div class="metric-label">Average Math Score</div>
          <div class="metric-value" id="mAvgMath">-</div>
          <div class="metric-sub">Scale: 0 - 100 points</div>
        </div>
        <div class="metric-card emerald">
          <div class="metric-label">Average Reading Score</div>
          <div class="metric-value" id="mAvgReading">-</div>
          <div class="metric-sub">Scale: 0 - 100 points</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Average Writing Score</div>
          <div class="metric-value" id="mAvgWriting">-</div>
          <div class="metric-sub">Scale: 0 - 100 points</div>
        </div>
        <div class="metric-card teal">
          <div class="metric-label">Data Quality</div>
          <div class="metric-value" id="mDataQuality">Clean</div>
          <div class="metric-sub" id="mDataQualitySub">0 missing, 0 duplicates</div>
        </div>
      </div>

      <!-- Educational Primer -->
      <div class="card">
        <div class="card-header">
          <h2>📖 Instructor Lesson: What is Exploratory Data Analysis (EDA)?</h2>
        </div>
        <p style="font-size: 0.95rem; color: #334155; margin-bottom: 16px;">
          <strong>Exploratory Data Analysis (EDA)</strong> is the foundational step in data science. Before training machine learning models or making business decisions, data scientists use summary statistics and graphical visualizations to understand the structure of the data, uncover patterns, test hypotheses, spot anomalies, and check assumptions.
        </p>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
          <div style="background: #F8FAFC; border: 1px solid var(--border); border-radius: 8px; padding: 14px;">
            <h4 style="color: var(--primary); font-size: 0.95rem; margin-bottom: 6px;">1. Univariate Analysis</h4>
            <p style="font-size: 0.88rem; color: #475569;">
              Examines <em>one single variable</em> at a time. Aims to reveal the distribution shape, central tendency (mean, median), spread (standard deviation, IQR), and identify outliers. Tools: Histograms, Box Plots, Count Plots.
            </p>
          </div>
          <div style="background: #F8FAFC; border: 1px solid var(--border); border-radius: 8px; padding: 14px;">
            <h4 style="color: var(--secondary); font-size: 0.95rem; margin-bottom: 6px;">2. Bivariate Analysis</h4>
            <p style="font-size: 0.88rem; color: #475569;">
              Explores relationships between <em>two variables</em>. Tests how one feature behaves as another changes (e.g., scores vs. lunch type). Tools: Scatter Plots, Grouped Bar Charts, Category Box Plots, Line Plots.
            </p>
          </div>
          <div style="background: #F8FAFC; border: 1px solid var(--border); border-radius: 8px; padding: 14px;">
            <h4 style="color: var(--accent); font-size: 0.95rem; margin-bottom: 6px;">3. Multivariate Analysis</h4>
            <p style="font-size: 0.88rem; color: #475569;">
              Investigates interactions across <em>three or more variables</em> concurrently. Uncovers collinearity, clustering, and interactions. Tools: Correlation Heatmaps, Pair Plots, Grouped Facet Grids.
            </p>
          </div>
        </div>
      </div>

      <!-- Dataset Preview Table -->
      <div class="card">
        <div class="card-header">
          <h2>🔍 Dataset Preview (First 5 Rows)</h2>
        </div>
        <div class="table-container">
          <table id="previewTable">
            <thead>
              <tr id="previewTableHead"><th>Loading table...</th></tr>
            </thead>
            <tbody id="previewTableBody">
            </tbody>
          </table>
        </div>
      </div>

      <!-- Descriptive Statistics Table -->
      <div class="card">
        <div class="card-header">
          <h2>📊 Summary Statistics for Numerical Variables (df.describe())</h2>
        </div>
        <div class="table-container">
          <table id="statsTable">
            <thead>
              <tr id="statsTableHead"><th>Statistic</th></tr>
            </thead>
            <tbody id="statsTableBody">
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- ===================================================================== -->
    <!-- TAB 2: UNIVARIATE ANALYSIS -->
    <!-- ===================================================================== -->
    <section id="tab-univariate" class="tab-panel">
      <div class="card">
        <div class="card-header">
          <h2>📊 Univariate Analysis: Single-Variable Exploration</h2>
        </div>

        <div class="toolbar">
          <div class="form-group">
            <label for="uniColumnSelect">Select Column</label>
            <select id="uniColumnSelect" onchange="onUnivariateColumnChange()">
              <!-- Options populated dynamically -->
            </select>
          </div>

          <div class="form-group">
            <label for="uniChartTypeSelect">Chart Type</label>
            <select id="uniChartTypeSelect" onchange="renderUnivariateChart()">
              <option value="histogram">Histogram & KDE (Continuous)</option>
              <option value="box_plot">Box Plot (Continuous)</option>
              <option value="count_plot">Count Plot (Categorical)</option>
              <option value="pie_chart">Pie Chart (Categorical)</option>
            </select>
          </div>

          <div class="form-group" style="align-self: flex-end;">
            <button class="btn btn-primary" onclick="renderUnivariateChart()">Generate Chart</button>
          </div>
        </div>

        <div class="chart-layout">
          <!-- Chart Display -->
          <div class="chart-box" id="uniChartBox">
            <div class="spinner" id="uniSpinner" style="display: none;"></div>
            <img id="uniChartImg" class="chart-img" alt="Univariate Chart" src="" />
          </div>

          <!-- Explanation & Code -->
          <div class="info-card">
            <div class="info-item">
              <h4 id="uniExpTitle">🎯 Chart Purpose</h4>
              <p id="uniExpPurpose">Select a column above to explore its univariate distribution.</p>
            </div>
            <div class="info-item">
              <h4>⏱️ When to Use</h4>
              <p id="uniExpWhen">Use this chart to observe frequency, skewness, and outliers.</p>
            </div>
            <div class="info-item">
              <h4>🔍 What to Observe in this Dataset</h4>
              <p id="uniExpObserve">Notice whether the data forms a normal bell-shaped curve or discrete groups.</p>
            </div>

            <!-- Code Section -->
            <div class="code-container">
              <div class="code-header">
                <div class="code-tabs">
                  <button class="code-tab-btn active" id="uniBtnSns" onclick="setUniCodeTab('seaborn')">Seaborn</button>
                  <button class="code-tab-btn" id="uniBtnMpl" onclick="setUniCodeTab('matplotlib')">Matplotlib</button>
                </div>
                <button class="copy-btn" onclick="copyCode('uniCodeBox')">📋 Copy</button>
              </div>
              <pre class="code-box" id="uniCodeBox"># Python Code</pre>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ===================================================================== -->
    <!-- TAB 3: BIVARIATE ANALYSIS -->
    <!-- ===================================================================== -->
    <section id="tab-bivariate" class="tab-panel">
      <div class="card">
        <div class="card-header">
          <h2>🔀 Bivariate Analysis: Two-Variable Relationships</h2>
        </div>

        <div class="toolbar">
          <div class="form-group">
            <label for="biPresetSelect">Classroom Example Presets</label>
            <select id="biPresetSelect" onchange="applyBiPreset()">
              <option value="custom">-- Choose a Preset or Custom --</option>
              <option value="p1">Average Math Score by Gender (Bar Chart)</option>
              <option value="p2">Math Score vs. Reading Score (Scatter Plot)</option>
              <option value="p3">Writing Score by Lunch Type (Box Plot)</option>
              <option value="p4">Parental Education vs. Writing Score (Line Chart)</option>
              <option value="p5">Test Preparation vs. Math Score (Bar Chart)</option>
            </select>
          </div>

          <div class="form-group">
            <label for="biXSelect">X Column (Category or Predictor)</label>
            <select id="biXSelect" onchange="renderBivariateChart()">
              <!-- Options populated dynamically -->
            </select>
          </div>

          <div class="form-group">
            <label for="biYSelect">Y Column (Numerical Target)</label>
            <select id="biYSelect" onchange="renderBivariateChart()">
              <!-- Options populated dynamically -->
            </select>
          </div>

          <div class="form-group">
            <label for="biChartTypeSelect">Chart Type</label>
            <select id="biChartTypeSelect" onchange="renderBivariateChart()">
              <option value="bar_chart">Bar Chart (Mean Comparison)</option>
              <option value="scatter_plot">Scatter Plot (Two Numerics)</option>
              <option value="box_plot">Grouped Box Plot (Distribution by Group)</option>
              <option value="line_chart">Line Chart (Ordered Trend)</option>
            </select>
          </div>

          <div class="form-group" style="align-self: flex-end;">
            <button class="btn btn-primary" onclick="renderBivariateChart()">Update Chart</button>
          </div>
        </div>

        <div class="chart-layout">
          <!-- Chart Display -->
          <div class="chart-box" id="biChartBox">
            <div class="spinner" id="biSpinner" style="display: none;"></div>
            <img id="biChartImg" class="chart-img" alt="Bivariate Chart" src="" />
          </div>

          <!-- Explanation & Code -->
          <div class="info-card">
            <div class="info-item">
              <h4 id="biExpTitle">🎯 Chart Purpose</h4>
              <p id="biExpPurpose">Examine how two features interact.</p>
            </div>
            <div class="info-item">
              <h4>⏱️ When to Use</h4>
              <p id="biExpWhen">Compare averages or test correlation between features.</p>
            </div>
            <div class="info-item">
              <h4>🔍 What to Observe in this Dataset</h4>
              <p id="biExpObserve">Notice the gaps between demographic subgroups.</p>
            </div>

            <!-- Code Section -->
            <div class="code-container">
              <div class="code-header">
                <div class="code-tabs">
                  <button class="code-tab-btn active" id="biBtnSns" onclick="setBiCodeTab('seaborn')">Seaborn</button>
                  <button class="code-tab-btn" id="biBtnMpl" onclick="setBiCodeTab('matplotlib')">Matplotlib</button>
                </div>
                <button class="copy-btn" onclick="copyCode('biCodeBox')">📋 Copy</button>
              </div>
              <pre class="code-box" id="biCodeBox"># Python Code</pre>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ===================================================================== -->
    <!-- TAB 4: MULTIVARIATE ANALYSIS -->
    <!-- ===================================================================== -->
    <section id="tab-multivariate" class="tab-panel">
      <div class="card">
        <div class="card-header">
          <h2>🌐 Multivariate Analysis: Multi-Variable Interactions</h2>
        </div>

        <div class="toolbar">
          <div class="form-group">
            <label>Select Multivariate Visualization</label>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
              <button class="btn btn-primary" onclick="renderMultivariate('heatmap')">🔥 Correlation Heatmap</button>
              <button class="btn btn-primary" onclick="renderMultivariate('pairplot')">🔲 Pair Plot Matrix</button>
              <button class="btn btn-primary" onclick="renderMultivariate('scatter_hue')">🎨 Scatter with Gender Hue</button>
            </div>
          </div>
        </div>

        <div class="chart-layout">
          <!-- Chart Display -->
          <div class="chart-box" id="multiChartBox">
            <div class="spinner" id="multiSpinner" style="display: none;"></div>
            <img id="multiChartImg" class="chart-img" alt="Multivariate Chart" src="" />
          </div>

          <!-- Explanation & Code -->
          <div class="info-card">
            <div class="info-item">
              <h4 id="multiExpTitle">🎯 Multivariate Focus</h4>
              <p id="multiExpPurpose">Analyzes multiple features simultaneously.</p>
            </div>
            <div class="info-item">
              <h4>⏱️ When to Use</h4>
              <p id="multiExpWhen">Uncover systemic collinearity and interaction effects across all exam subjects.</p>
            </div>
            <div class="info-item">
              <h4>🔍 What to Observe in this Dataset</h4>
              <p id="multiExpObserve">Notice the strong positive correlation between reading and writing (r > 0.90).</p>
            </div>

            <!-- Code Section -->
            <div class="code-container">
              <div class="code-header">
                <div class="code-tabs">
                  <button class="code-tab-btn active" id="multiBtnSns" onclick="setMultiCodeTab('seaborn')">Seaborn</button>
                  <button class="code-tab-btn" id="multiBtnMpl" onclick="setMultiCodeTab('matplotlib')">Matplotlib</button>
                </div>
                <button class="copy-btn" onclick="copyCode('multiCodeBox')">📋 Copy</button>
              </div>
              <pre class="code-box" id="multiCodeBox"># Python Code</pre>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ===================================================================== -->
    <!-- TAB 5: MATPLOTLIB VS SEABORN COMPARISON -->
    <!-- ===================================================================== -->
    <section id="tab-comparison" class="tab-panel">
      <div class="card">
        <div class="card-header">
          <h2>⚔️ Matplotlib vs. Seaborn: Classroom Side-by-Side Guide</h2>
        </div>

        <div class="callout-box info">
          <h4>💡 The Core Difference for Beginners</h4>
          <p>
            <strong>Matplotlib</strong> is a <em>low-level, imperative</em> library. It gives you absolute, fine-grained control over every pixel, axis, and tick, but you must manually calculate aggregations, compute coordinates, and configure legends.<br>
            <strong>Seaborn</strong> is a <em>high-level, declarative</em> statistical library built on top of Matplotlib. It directly understands Pandas DataFrames, performs automatic statistical computations (means, confidence intervals, KDE curves), and applies attractive styling with concise code.
          </p>
        </div>

        <div class="toolbar">
          <div class="form-group">
            <label>Select Chart to Compare Syntax</label>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
              <button class="btn btn-primary" onclick="loadComparison('histogram')">Histogram</button>
              <button class="btn btn-primary" onclick="loadComparison('bar_chart')">Bar Chart</button>
              <button class="btn btn-primary" onclick="loadComparison('box_plot')">Box Plot</button>
              <button class="btn btn-primary" onclick="loadComparison('scatter_plot')">Scatter Plot</button>
              <button class="btn btn-primary" onclick="loadComparison('heatmap')">Heatmap</button>
            </div>
          </div>
        </div>

        <h3 id="compChartTitle" style="font-size: 1.15rem; color: var(--primary-dark); margin: 12px 0 6px;">
          Comparing: Histogram Syntax
        </h3>
        <p id="compInsightText" style="font-size: 0.9rem; color: #475569; margin-bottom: 16px;">
          Notice how Seaborn automatically estimates and overlays the KDE probability curve with a single flag (kde=True), whereas Matplotlib requires manual computation or external libraries.
        </p>

        <div class="comparison-grid">
          <!-- Matplotlib Column -->
          <div class="code-container" style="margin-top: 0;">
            <div class="code-header" style="background: #1E293B;">
              <span style="color: #60A5FA; font-weight: 700; font-size: 0.85rem;">📊 Matplotlib (Low-Level Control)</span>
              <button class="copy-btn" onclick="copyCode('compMplCode')">📋 Copy</button>
            </div>
            <pre class="code-box" id="compMplCode" style="max-height: 380px;"># Matplotlib code loading...</pre>
          </div>

          <!-- Seaborn Column -->
          <div class="code-container" style="margin-top: 0;">
            <div class="code-header" style="background: #1E293B;">
              <span style="color: #34D399; font-weight: 700; font-size: 0.85rem;">🎨 Seaborn (High-Level Statistical)</span>
              <button class="copy-btn" onclick="copyCode('compSnsCode')">📋 Copy</button>
            </div>
            <pre class="code-box" id="compSnsCode" style="max-height: 380px;"># Seaborn code loading...</pre>
          </div>
        </div>

        <!-- Philosophical Comparison Table -->
        <div style="margin-top: 24px;">
          <h3 style="font-size: 1.1rem; color: #1E293B; margin-bottom: 12px;">Summary: When to Choose Which?</h3>
          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th>Aspect</th>
                  <th>Matplotlib</th>
                  <th>Seaborn</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Programming Style</strong></td>
                  <td>Imperative (tell the computer every step)</td>
                  <td>Declarative (tell the computer what you want to see)</td>
                </tr>
                <tr>
                  <td><strong>Pandas Integration</strong></td>
                  <td>Requires passing raw Series/arrays or pre-aggregating</td>
                  <td>Native integration; passes <code>data=df, x='col', y='col'</code></td>
                </tr>
                <tr>
                  <td><strong>Statistical Calculations</strong></td>
                  <td>Manual calculation needed for error bars, KDE, regression</td>
                  <td>Automatic estimation (KDE, 95% confidence intervals, regression lines)</td>
                </tr>
                <tr>
                  <td><strong>Grouping & Legends</strong></td>
                  <td>Requires custom loops and manual Legend handles</td>
                  <td>Automatic grouping with <code>hue='category'</code></td>
                </tr>
                <tr>
                  <td><strong>Best Used For</strong></td>
                  <td>Fine-tuning figures for publication, custom subplots, dashboards</td>
                  <td>Rapid exploratory data analysis (EDA), statistical storytelling</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>

    <!-- ===================================================================== -->
    <!-- TAB 6: GRAPH COMPARISON GUIDE -->
    <!-- ===================================================================== -->
    <section id="tab-guide" class="tab-panel">
      <div class="card">
        <div class="card-header">
          <h2>📑 Classroom Reference: The 9 Essential EDA Graph Types</h2>
        </div>
        <p style="font-size: 0.95rem; color: #475569; margin-bottom: 16px;">
          This reference table summarizes the purpose, suitable data types, example questions, and limitations for each of the 9 foundational exploratory graphs.
        </p>

        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Graph Name</th>
                <th>Primary Purpose</th>
                <th>When to Use</th>
                <th>Example Question Answered</th>
                <th>Data Type</th>
                <th>Limitations</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>1. Bar Chart</strong></td>
                <td>Compare numerical aggregations across discrete groups</td>
                <td>Comparing means, sums, or rates across categories</td>
                <td><em>"What is the average math score by gender?"</em></td>
                <td>1 Categorical + 1 Numerical</td>
                <td>Hides variance and outliers (only shows mean)</td>
              </tr>
              <tr>
                <td><strong>2. Line Chart</strong></td>
                <td>Show ordered trends or progressions</td>
                <td>When the X-axis represents an ordinal progression or time</td>
                <td><em>"Do scores rise monotonically as parental education levels increase?"</em></td>
                <td>1 Ordered Categorical + 1 Numerical</td>
                <td>Misleading if X categories have no inherent order</td>
              </tr>
              <tr>
                <td><strong>3. Histogram</strong></td>
                <td>Examine distribution shape, skewness, and spread</td>
                <td>Inspecting continuous numerical variables for normality</td>
                <td><em>"Are student math scores normally distributed?"</em></td>
                <td>1 Continuous Numerical</td>
                <td>Bin width choices can obscure distribution shape</td>
              </tr>
              <tr>
                <td><strong>4. Box Plot</strong></td>
                <td>Display 5-number summary (median, IQR) and spot outliers</td>
                <td>Comparing distributions and identifying extreme values</td>
                <td><em>"Which subject has the most low-score outliers?"</em></td>
                <td>1 Numerical (+ Optional 1 Categorical)</td>
                <td>Does not show sample size or bimodal shapes</td>
              </tr>
              <tr>
                <td><strong>5. Scatter Plot</strong></td>
                <td>Detect correlation, linearity, and clustering</td>
                <td>Examining association between two continuous measurements</td>
                <td><em>"Does reading performance strongly predict writing score?"</em></td>
                <td>2 Continuous Numerical (+ Optional Categorical Hue)</td>
                <td>Overplotting occurs with large sample sizes; doesn't prove causation</td>
              </tr>
              <tr>
                <td><strong>6. Heatmap</strong></td>
                <td>Display pairwise correlation matrix via color intensity</td>
                <td>Scanning all numerical variables simultaneously for relationships</td>
                <td><em>"Which pair of exam subjects is most tightly correlated?"</em></td>
                <td>Matrix of Numerical Columns</td>
                <td>Only captures linear correlation (Pearson r)</td>
              </tr>
              <tr>
                <td><strong>7. Count Plot</strong></td>
                <td>Display frequencies of discrete categories</td>
                <td>Assessing sample balance and demographic counts</td>
                <td><em>"How many students belong to each race/ethnic group?"</em></td>
                <td>1 Categorical</td>
                <td>Only shows frequencies, not relationships with outcomes</td>
              </tr>
              <tr>
                <td><strong>8. Pie Chart</strong></td>
                <td>Show relative percentage share of a whole</td>
                <td>Comparing 2 to 5 distinct categories when proportional share matters</td>
                <td><em>"What percentage of students receive free/reduced lunch?"</em></td>
                <td>1 Categorical (2–5 categories)</td>
                <td>Difficult to accurately compare slice angles; avoid when >5 groups</td>
              </tr>
              <tr>
                <td><strong>9. Pair Plot</strong></td>
                <td>Visualize all pairwise scatter plots and univariate distributions</td>
                <td>Initial holistic scan of all continuous features in a dataset</td>
                <td><em>"How do math, reading, and writing interact simultaneously?"</em></td>
                <td>3+ Continuous Numerical (+ Optional Categorical Hue)</td>
                <td>Computationally heavy; can become cluttered with many columns</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </div>

  <!-- Client-Side JavaScript -->
  <script>
    // State management
    let datasetSummary = null;
    let currentUniCode = { seaborn: '', matplotlib: '' };
    let currentBiCode = { seaborn: '', matplotlib: '' };
    let currentMultiCode = { seaborn: '', matplotlib: '' };
    let activeUniTab = 'seaborn';
    let activeBiTab = 'seaborn';
    let activeMultiTab = 'seaborn';

    // Initialization on DOM ready
    window.addEventListener('DOMContentLoaded', () => {
      fetchSummary();
      loadComparison('histogram');
    });

    // Tab Switching Function
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
      
      const targetPanel = document.getElementById('tab-' + tabId);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }
      
      const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
      if (activeBtn) activeBtn.classList.add('active');

      // Trigger default render if opening tab for first time
      if (tabId === 'univariate' && !document.getElementById('uniChartImg').src) {
        renderUnivariateChart();
      } else if (tabId === 'bivariate' && !document.getElementById('biChartImg').src) {
        renderBivariateChart();
      } else if (tabId === 'multivariate' && !document.getElementById('multiChartImg').src) {
        renderMultivariate('heatmap');
      }
    }

    // Fetch Dataset Summary from Flask API
    function fetchSummary() {
      fetch('/api/summary')
        .then(res => res.json())
        .then(data => {
          if (data.status === 'error') {
            const errBanner = document.getElementById('errorBanner');
            errBanner.style.display = 'block';
            errBanner.innerHTML = '<strong>⚠️ Dataset Error:</strong> ' + data.message + 
              '<br><small>Please download <code>StudentsPerformance.csv</code> from Kaggle and place it in the same directory as <code>eda_dashboard.py</code>.</small>';
            return;
          }
          datasetSummary = data;
          populateOverview(data);
          populateDropdowns(data);
          // Auto-render default univariate chart
          renderUnivariateChart();
        })
        .catch(err => {
          console.error("Error fetching summary:", err);
          const errBanner = document.getElementById('errorBanner');
          errBanner.style.display = 'block';
          errBanner.innerText = 'Failed to connect to backend Flask API.';
        });
    }

    // Populate Overview Tab
    function populateOverview(data) {
      document.getElementById('mTotalStudents').innerText = data.total_students.toLocaleString();
      document.getElementById('mTotalColumns').innerText = data.total_columns;
      document.getElementById('mAvgMath').innerText = data.avg_math_score;
      document.getElementById('mAvgReading').innerText = data.avg_reading_score;
      document.getElementById('mAvgWriting').innerText = data.avg_writing_score;
      document.getElementById('mDataQualitySub').innerText = 
        `${data.total_missing} missing values, ${data.duplicate_count} duplicates`;
      document.getElementById('headerStudentCount').innerText = `(${data.total_students.toLocaleString()} Students)`;

      // Populate preview table
      if (data.preview_data && data.preview_data.length > 0) {
        const headRow = document.getElementById('previewTableHead');
        headRow.innerHTML = '';
        const cols = Object.keys(data.preview_data[0]);
        cols.forEach(c => {
          const th = document.createElement('th');
          th.innerText = c;
          headRow.appendChild(th);
        });

        const tbody = document.getElementById('previewTableBody');
        tbody.innerHTML = '';
        data.preview_data.forEach(row => {
          const tr = document.createElement('tr');
          cols.forEach(c => {
            const td = document.createElement('td');
            td.innerText = row[c];
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
      }

      // Populate statistics table
      if (data.summary_stats && data.summary_stats.length > 0) {
        const headRow = document.getElementById('statsTableHead');
        headRow.innerHTML = '<th>Stat Metric</th>';
        const numCols = data.numeric_columns;
        numCols.forEach(c => {
          const th = document.createElement('th');
          th.innerText = c;
          headRow.appendChild(th);
        });

        const tbody = document.getElementById('statsTableBody');
        tbody.innerHTML = '';
        data.summary_stats.forEach(row => {
          const tr = document.createElement('tr');
          const tdStat = document.createElement('td');
          tdStat.innerHTML = `<strong>${row['index'] || ''}</strong>`;
          tr.appendChild(tdStat);
          numCols.forEach(c => {
            const td = document.createElement('td');
            td.innerText = row[c] !== undefined ? row[c] : '-';
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
      }
    }

    // Populate Dropdown Menus
    function populateDropdowns(data) {
      const uniSelect = document.getElementById('uniColumnSelect');
      const biXSelect = document.getElementById('biXSelect');
      const biYSelect = document.getElementById('biYSelect');

      uniSelect.innerHTML = '';
      biXSelect.innerHTML = '';
      biYSelect.innerHTML = '';

      // Univariate column grouping
      const numOptGroup = document.createElement('optgroup');
      numOptGroup.label = "Numerical Columns";
      data.numeric_columns.forEach(col => {
        const opt = document.createElement('option');
        opt.value = col;
        opt.innerText = col;
        numOptGroup.appendChild(opt);
      });
      uniSelect.appendChild(numOptGroup);

      const catOptGroup = document.createElement('optgroup');
      catOptGroup.label = "Categorical Columns";
      data.categorical_columns.forEach(col => {
        const opt = document.createElement('option');
        opt.value = col;
        opt.innerText = col;
        catOptGroup.appendChild(opt);
      });
      uniSelect.appendChild(catOptGroup);

      // Bivariate X options (all columns)
      data.columns.forEach(col => {
        const opt = document.createElement('option');
        opt.value = col;
        opt.innerText = col;
        if (col === 'gender') opt.selected = true;
        biXSelect.appendChild(opt);
      });

      // Bivariate Y options (numeric columns)
      data.numeric_columns.forEach(col => {
        const opt = document.createElement('option');
        opt.value = col;
        opt.innerText = col;
        if (col === 'math score') opt.selected = true;
        biYSelect.appendChild(opt);
      });
    }

    // Univariate Column Selection Change Handler
    function onUnivariateColumnChange() {
      const col = document.getElementById('uniColumnSelect').value;
      const chartSelect = document.getElementById('uniChartTypeSelect');
      if (datasetSummary) {
        if (datasetSummary.numeric_columns.includes(col)) {
          chartSelect.value = 'histogram';
        } else {
          chartSelect.value = 'count_plot';
        }
      }
      renderUnivariateChart();
    }

    // Render Univariate Chart via Flask API
    function renderUnivariateChart() {
      const col = document.getElementById('uniColumnSelect').value;
      const chartType = document.getElementById('uniChartTypeSelect').value;
      if (!col) return;

      const img = document.getElementById('uniChartImg');
      const spinner = document.getElementById('uniSpinner');
      spinner.style.display = 'block';
      img.style.opacity = '0.3';

      fetch(`/api/chart?type=${chartType}&x=${encodeURIComponent(col)}`)
        .then(res => res.json())
        .then(data => {
          spinner.style.display = 'none';
          img.style.opacity = '1';
          if (data.image) {
            img.src = data.image;
          }
          if (data.explanation) {
            document.getElementById('uniExpTitle').innerText = '🎯 ' + data.explanation.title;
            document.getElementById('uniExpPurpose').innerText = data.explanation.purpose;
            document.getElementById('uniExpWhen').innerText = data.explanation.when_to_use;
            document.getElementById('uniExpObserve').innerText = data.explanation.what_to_observe;
          }
          currentUniCode.seaborn = data.seaborn_code || '';
          currentUniCode.matplotlib = data.matplotlib_code || '';
          updateUniCodeBox();
        })
        .catch(err => {
          spinner.style.display = 'none';
          console.error("Error rendering univariate chart:", err);
        });
    }

    function setUniCodeTab(type) {
      activeUniTab = type;
      document.getElementById('uniBtnSns').classList.toggle('active', type === 'seaborn');
      document.getElementById('uniBtnMpl').classList.toggle('active', type === 'matplotlib');
      updateUniCodeBox();
    }
    function updateUniCodeBox() {
      document.getElementById('uniCodeBox').innerText = currentUniCode[activeUniTab] || '# No code';
    }

    // Apply Bivariate Classroom Preset
    function applyBiPreset() {
      const preset = document.getElementById('biPresetSelect').value;
      const xSel = document.getElementById('biXSelect');
      const ySel = document.getElementById('biYSelect');
      const typeSel = document.getElementById('biChartTypeSelect');

      if (preset === 'p1') {
        xSel.value = 'gender';
        ySel.value = 'math score';
        typeSel.value = 'bar_chart';
      } else if (preset === 'p2') {
        xSel.value = 'math score';
        ySel.value = 'reading score';
        typeSel.value = 'scatter_plot';
      } else if (preset === 'p3') {
        xSel.value = 'lunch';
        ySel.value = 'writing score';
        typeSel.value = 'box_plot';
      } else if (preset === 'p4') {
        xSel.value = 'parental level of education';
        ySel.value = 'writing score';
        typeSel.value = 'line_chart';
      } else if (preset === 'p5') {
        xSel.value = 'test preparation course';
        ySel.value = 'math score';
        typeSel.value = 'bar_chart';
      }
      renderBivariateChart();
    }

    // Render Bivariate Chart via Flask API
    function renderBivariateChart() {
      const xCol = document.getElementById('biXSelect').value;
      const yCol = document.getElementById('biYSelect').value;
      const chartType = document.getElementById('biChartTypeSelect').value;
      if (!xCol || !yCol) return;

      const img = document.getElementById('biChartImg');
      const spinner = document.getElementById('biSpinner');
      spinner.style.display = 'block';
      img.style.opacity = '0.3';

      fetch(`/api/chart?type=${chartType}&x=${encodeURIComponent(xCol)}&y=${encodeURIComponent(yCol)}`)
        .then(res => res.json())
        .then(data => {
          spinner.style.display = 'none';
          img.style.opacity = '1';
          if (data.image) {
            img.src = data.image;
          }
          if (data.explanation) {
            document.getElementById('biExpTitle').innerText = '🎯 ' + data.explanation.title;
            document.getElementById('biExpPurpose').innerText = data.explanation.purpose;
            document.getElementById('biExpWhen').innerText = data.explanation.when_to_use;
            document.getElementById('biExpObserve').innerText = data.explanation.what_to_observe;
          }
          currentBiCode.seaborn = data.seaborn_code || '';
          currentBiCode.matplotlib = data.matplotlib_code || '';
          updateBiCodeBox();
        })
        .catch(err => {
          spinner.style.display = 'none';
          console.error("Error rendering bivariate chart:", err);
        });
    }

    function setBiCodeTab(type) {
      activeBiTab = type;
      document.getElementById('biBtnSns').classList.toggle('active', type === 'seaborn');
      document.getElementById('biBtnMpl').classList.toggle('active', type === 'matplotlib');
      updateBiCodeBox();
    }
    function updateBiCodeBox() {
      document.getElementById('biCodeBox').innerText = currentBiCode[activeBiTab] || '# No code';
    }

    // Render Multivariate Visualization
    function renderMultivariate(kind) {
      const img = document.getElementById('multiChartImg');
      const spinner = document.getElementById('multiSpinner');
      spinner.style.display = 'block';
      img.style.opacity = '0.3';

      let url = '/api/chart?type=heatmap';
      if (kind === 'pairplot') {
        url = '/api/chart?type=pairplot';
      } else if (kind === 'scatter_hue') {
        url = '/api/chart?type=scatter_plot&x=reading%20score&y=writing%20score&hue=gender';
      }

      fetch(url)
        .then(res => res.json())
        .then(data => {
          spinner.style.display = 'none';
          img.style.opacity = '1';
          if (data.image) {
            img.src = data.image;
          }
          if (data.explanation) {
            document.getElementById('multiExpTitle').innerText = '🎯 ' + data.explanation.title;
            document.getElementById('multiExpPurpose').innerText = data.explanation.purpose;
            document.getElementById('multiExpWhen').innerText = data.explanation.when_to_use;
            document.getElementById('multiExpObserve').innerText = data.explanation.what_to_observe;
          }
          currentMultiCode.seaborn = data.seaborn_code || '';
          currentMultiCode.matplotlib = data.matplotlib_code || '';
          updateMultiCodeBox();
        })
        .catch(err => {
          spinner.style.display = 'none';
          console.error("Error rendering multivariate chart:", err);
        });
    }

    function setMultiCodeTab(type) {
      activeMultiTab = type;
      document.getElementById('multiBtnSns').classList.toggle('active', type === 'seaborn');
      document.getElementById('multiBtnMpl').classList.toggle('active', type === 'matplotlib');
      updateMultiCodeBox();
    }
    function updateMultiCodeBox() {
      document.getElementById('multiCodeBox').innerText = currentMultiCode[activeMultiTab] || '# No code';
    }

    // Load Side-by-Side Comparison Code
    function loadComparison(chartType) {
      const titleElem = document.getElementById('compChartTitle');
      const insightElem = document.getElementById('compInsightText');
      const mplBox = document.getElementById('compMplCode');
      const snsBox = document.getElementById('compSnsCode');

      titleElem.innerText = `Comparing: ${chartType.replace('_', ' ').toUpperCase()} Syntax`;

      const insights = {
        histogram: "Seaborn estimates and renders the KDE curve in one shot (kde=True). Matplotlib requires plotting raw bins and using scipy or manual line math.",
        bar_chart: "Seaborn automatically aggregates data (mean by default) directly from the DataFrame. In Matplotlib, you must call df.groupby() and pass the grouped index and values.",
        box_plot: "Seaborn splits categories seamlessly with x='category' and y='score'. In Matplotlib, you must manually split data into a list of series or arrays.",
        scatter_plot: "In Seaborn, adding hue='gender' generates colors and a polished legend automatically. In Matplotlib, you must map categories to colors and construct custom legend handles.",
        heatmap: "Seaborn draws the annotated correlation grid in one line with annot=True. In Matplotlib, you must create a double for-loop to write text in every cell."
      };
      insightElem.innerText = insights[chartType] || "Notice the code verbosity and level of abstraction differences.";

      fetch(`/api/compare-code?type=${chartType}`)
        .then(res => res.json())
        .then(data => {
          mplBox.innerText = data.matplotlib_code;
          snsBox.innerText = data.seaborn_code;
        })
        .catch(err => {
          console.error("Error loading comparison code:", err);
        });
    }

    // Clipboard Copy Helper
    function copyCode(elemId) {
      const codeText = document.getElementById(elemId).innerText;
      navigator.clipboard.writeText(codeText).then(() => {
        alert("Code snippet copied to clipboard!");
      }).catch(err => {
        console.error("Failed to copy code: ", err);
      });
    }
  </script>
</body>
</html>
"""

# ==============================================================================
# 19. CREATE FLASK ROUTES
# ==============================================================================
def create_flask_routes(app):
    """Registers all necessary Flask routes to serve HTML and JSON API responses."""

    @app.route('/')
    def index():
        return render_template_string(create_dashboard_html())

    @app.route('/api/summary')
    def api_summary():
        df = load_dataset(CSV_PATH)
        df_clean = clean_dataset(df)
        summary = get_dataset_summary(df_clean)
        return jsonify(summary)

    @app.route('/api/chart')
    def api_chart():
        df = load_dataset(CSV_PATH)
        df_clean = clean_dataset(df)
        if df_clean is None:
            return jsonify({"error": f"Dataset file '{CSV_PATH}' not found."}), 404

        chart_type = request.args.get('type', 'histogram').lower()
        x_col = request.args.get('x', 'math score')
        y_col = request.args.get('y', 'reading score')
        hue_col = request.args.get('hue', None)

        img_b64 = None
        try:
            if chart_type == 'histogram':
                img_b64 = create_histogram(df_clean, x_col)
            elif chart_type == 'bar_chart':
                img_b64 = create_bar_chart(df_clean, x_col, y_col)
            elif chart_type == 'line_chart':
                img_b64 = create_line_chart(df_clean, x_col, y_col)
            elif chart_type == 'box_plot':
                by_col = x_col if x_col != y_col and x_col in get_categorical_columns(df_clean) else None
                val_col = y_col if by_col else x_col
                img_b64 = create_box_plot(df_clean, val_col, by_column=by_col)
            elif chart_type == 'scatter_plot':
                img_b64 = create_scatter_plot(df_clean, x_col, y_col, hue_column=hue_col)
            elif chart_type == 'heatmap':
                img_b64 = create_heatmap(df_clean)
            elif chart_type == 'count_plot':
                img_b64 = create_count_plot(df_clean, x_col)
            elif chart_type == 'pie_chart':
                img_b64 = create_pie_chart(df_clean, x_col)
            elif chart_type == 'pairplot':
                img_b64 = create_pairplot(df_clean)
            else:
                return jsonify({"error": f"Unknown chart type: {chart_type}"}), 400

            explanation = get_chart_explanation(chart_type, x_col, y_col)
            mpl_code = generate_matplotlib_code(chart_type, x_col, y_col, hue_col)
            sns_code = generate_seaborn_code(chart_type, x_col, y_col, hue_col)

            return jsonify({
                "chart_type": chart_type,
                "image": img_b64,
                "explanation": explanation,
                "matplotlib_code": mpl_code,
                "seaborn_code": sns_code
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/compare-code')
    def api_compare_code():
        chart_type = request.args.get('type', 'histogram').lower()
        mpl_code = generate_matplotlib_code(chart_type)
        sns_code = generate_seaborn_code(chart_type)
        return jsonify({
            "chart_type": chart_type,
            "matplotlib_code": mpl_code,
            "seaborn_code": sns_code
        })

# ==============================================================================
# 20. MAIN FUNCTION
# ==============================================================================
def main():
    """Initializes and runs the Flask application."""
    app = Flask(__name__)
    create_flask_routes(app)
    
    print("=" * 70)
    print("[*] EDA Classroom Dashboard (Single-File Architecture)")
    print(f"Dataset path: {os.path.abspath(CSV_PATH)}")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    print("=" * 70)
    
    app.run(debug=True, host="127.0.0.1", port=5000)

if __name__ == '__main__':
    main()
