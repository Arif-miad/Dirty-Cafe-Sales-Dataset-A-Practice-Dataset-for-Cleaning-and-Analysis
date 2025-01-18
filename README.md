# Dirty Cafe Sales Dataset: A Practice Dataset for Cleaning and Analysis

## Overview
The **Dirty Cafe Sales Dataset** offers a realistic and challenging dataset for practicing essential data science skills. It contains 10,000 rows of synthetic sales transaction data, designed with missing values, inconsistent data, and errors. This dataset is ideal for learning and applying data cleaning, exploratory data analysis (EDA), feature engineering, and machine learning techniques.

---

## File Information
- **File Name**: `dirty_cafe_sales.csv`
- **Number of Rows**: 10,000
- **Number of Columns**: 8

---

## Dataset Columns
1. **Transaction_ID**: Unique identifier for each transaction.
2. **Date**: Date of the transaction.
3. **Product**: Name of the product sold.
4. **Price**: Price of the product (contains some errors and inconsistencies).
5. **Quantity**: Number of products sold (may include missing values).
6. **Total_Sales**: Computed sales amount (some errors may exist).
7. **Payment_Method**: Payment method used (e.g., Cash, Credit Card).
8. **Customer_Age**: Age of the customer (may contain outliers).

---

## Potential Applications

1. **Data Cleaning**
   - Handle missing values and correct inconsistencies.
   - Standardize categorical data.
   - Address erroneous numerical entries.

2. **Exploratory Data Analysis (EDA)**
   - Analyze data distributions, patterns, and trends.
   - Visualize insights through charts and graphs.

3. **Feature Engineering**
   - Generate new features like sales per customer or discounts applied.
   - Prepare the dataset for machine learning models.

4. **Predictive Modeling**
   - Build regression models to predict sales.
   - Classify transactions based on specific attributes (e.g., payment methods).

---

## Code Implementation
Here’s an example pipeline for working with the dataset:

### 1. Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
```

### 2. Load Dataset
```python
# Load the dataset
df = pd.read_csv('dirty_cafe_sales.csv')

# View the first few rows
print(df.head())
```

### 3. Handle Missing Values
```python
# Fill missing numerical values with the mean
num_imputer = SimpleImputer(strategy='mean')
df['Price'] = num_imputer.fit_transform(df[['Price']])
df['Quantity'] = num_imputer.fit_transform(df[['Quantity']])

# Fill missing categorical values with 'Unknown'
df['Product'].fillna('Unknown', inplace=True)
df['Payment_Method'].fillna('Unknown', inplace=True)
```

### 4. Data Cleaning
```python
# Remove outliers in 'Customer_Age'
df = df[(df['Customer_Age'] >= 18) & (df['Customer_Age'] <= 100)]

# Correct inconsistencies in 'Payment_Method'
df['Payment_Method'] = df['Payment_Method'].str.strip().str.lower()
df['Payment_Method'] = df['Payment_Method'].replace({'creditcard': 'credit card'})
```

### 5. Exploratory Data Analysis
```python
# Plot sales distribution
sns.histplot(df['Total_Sales'], kde=True, color='blue')
plt.title('Distribution of Total Sales')
plt.show()

# Analyze sales trends over time
df['Date'] = pd.to_datetime(df['Date'])
df.groupby(df['Date'].dt.to_period('M'))['Total_Sales'].sum().plot(kind='line')
plt.title('Monthly Sales Trends')
plt.show()
```

### 6. Feature Engineering
```python
# Create a new feature: Average Sale per Product
df['Avg_Sale_Per_Product'] = df['Total_Sales'] / df['Quantity']

# Encode categorical variables
encoder = LabelEncoder()
df['Product'] = encoder.fit_transform(df['Product'])
df['Payment_Method'] = encoder.fit_transform(df['Payment_Method'])
```

### 7. Model Training
```python
# Prepare features and target
X = df[['Price', 'Quantity', 'Customer_Age', 'Product']]
y = df['Total_Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse:.2f}')
```

### 8. Save Cleaned Dataset
```python
# Save the cleaned dataset
df.to_csv('cleaned_cafe_sales.csv', index=False)
print("Cleaned dataset saved as 'cleaned_cafe_sales.csv'")
```





