import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Task 1: Data Cleaning
# -----------------------------
data = pd.read_csv("StudentsPerformance.csv")
print("First 5 rows of raw data:")
print(data.head())

# Check basic info
print("\nDataset Information:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())
print(f"\nDataset Shape: {data.shape}")

# Remove duplicate entries
duplicates = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
data = data.drop_duplicates()

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values (example column)
# Replace 'column_name' with actual column if needed
# data['column_name'] = data['column_name'].fillna(data['column_name'].mean())

# Drop rows with missing values
data = data.dropna()

# Fix formatting and data types
# Convert to datetime if applicable
# data['date_column'] = pd.to_datetime(data['date_column'], errors='coerce')

# Remove unwanted spaces and standardize text
# Example columns, replace if needed
# data['name'] = data['name'].str.strip()
# data['category'] = data['category'].str.lower()

# Save cleaned data
data.to_csv("cleaned_data.csv", index=False)
print("\nCleaned data saved as 'cleaned_data.csv'.")

# -----------------------------
# Task 2: Exploratory Data Analysis (EDA)
# -----------------------------
print("\n--- Exploratory Data Analysis (EDA) ---")

# Load cleaned dataset
data = pd.read_csv("cleaned_data.csv")

# Calculate basic statistics
print("\nBasic Statistics:")
print("Mean values:\n", data.mean(numeric_only=True))
print("\nMedian values:\n", data.median(numeric_only=True))
print("\nMaximum values:\n", data.max(numeric_only=True))
print("\nMinimum values:\n", data.min(numeric_only=True))

# Correlation matrix
print("\nCorrelation Matrix:")
correlation_matrix = data.corr(numeric_only=True)
print(correlation_matrix)

# Detect outliers using IQR
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))]
    print(f"\nOutliers in '{col}': {len(outliers)} rows")

# -----------------------------
# Visualizations
# -----------------------------
sns.set(style="whitegrid")

# Histogram for numeric columns
data.hist(figsize=(10, 8), bins=20)
plt.suptitle("Distribution of Numeric Features", fontsize=14)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Boxplot to detect outliers visually
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[numeric_cols])
plt.title("Boxplot of Numeric Features")
plt.show()

# Example: Bar chart for categorical column (uncomment if needed)
# sns.countplot(x='gender', data=data)
# plt.title("Distribution by Gender")
# plt.show()

# -----------------------------
# Automated Insights
# -----------------------------
print("\n--- Automated Insights ---")

# 1. Top correlated feature pairs
corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1]  # remove self-correlations
top_corr = corr_pairs.head(3)
print("\nTop 3 strongest correlations:")
print(top_corr)

# 2. Column with the highest mean
highest_mean_col = data.mean(numeric_only=True).idxmax()
print(f"\nColumn with highest average value: '{highest_mean_col}' -> {data[highest_mean_col].mean():.2f}")

# 3. Column with the largest variation
highest_std_col = data.std(numeric_only=True).idxmax()
print(f"Column with highest variability: '{highest_std_col}' -> Std Dev = {data[highest_std_col].std():.2f}")

# 4. General summary
print("\nSummary Insight:")
print("The dataset shows meaningful correlations and variability across key numerical features, providing a strong base for predictive or descriptive analysis.")

print("\nEDA completed successfully.")
