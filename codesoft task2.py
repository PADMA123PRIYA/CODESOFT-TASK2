# Step 1: Import libraries
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the CSV file 
df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')

# Show the first 5 rows
print(df.head())



# Step 2: Clean and preprocess the data
df = df.dropna(subset=['Rating'])

# Extract Year from 'Name' column if 'Year' is NaN
df['Extracted_Year'] = df['Name'].str.extract(r'\((\d{4})\)').astype(float)
df['Year'] = df['Year'].fillna(df['Extracted_Year'])
df.drop(columns=['Extracted_Year'], inplace=True)

df['Duration'] = df['Duration'].str.extract(r'(\d+)').astype(float)

for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col] = df[col].fillna("Unknown")

df['Votes'] = df['Votes'].astype(str).str.replace(',', '').replace('nan', '0').astype(int)


df['Main Genre'] = df['Genre'].str.split(',').str[0]

# Show the first 5 cleaned rows
print(df[['Name', 'Year', 'Duration', 'Main Genre', 'Director', 'Actor 1', 'Votes', 'Rating']].head())


# Step 3: Encode categorical features

from sklearn.preprocessing import LabelEncoder
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float)
le = LabelEncoder()

# Encode all relevant categorical columns
df['Director'] = le.fit_transform(df['Director'])
df['Actor 1'] = le.fit_transform(df['Actor 1'])
df['Actor 2'] = le.fit_transform(df['Actor 2'])
df['Actor 3'] = le.fit_transform(df['Actor 3'])
df['Main Genre'] = le.fit_transform(df['Main Genre'])
print(df[['Year', 'Duration', 'Director', 'Actor 1', 'Main Genre', 'Votes', 'Rating']].head())


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Step 4: Select features and target
features = ['Year', 'Duration', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Main Genre']
X = df[features]
y = df['Rating']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Show first 5 predictions alongside actual ratings
results = X_test.copy()
results['Actual Rating'] = y_test
results['Predicted Rating'] = predictions
print(results[['Actual Rating', 'Predicted Rating']].head())



#step:5
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

#Define features
features = ['Year', 'Duration', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Main Genre']
X = df[features]
y = df['Rating']
data_combined = pd.concat([X, y], axis=1)
data_cleaned = data_combined.dropna()

# Split clean data
X = data_cleaned[features]
y = data_cleaned['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Define and evaluate models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "XGBoost": XGBRegressor()
}

print("Model Performance (RMSE):\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name}: RMSE = {rmse:.3f}")


#step 6
best_model = RandomForestRegressor()
best_model.fit(X_train, y_train)

# Predict
predicted_ratings = best_model.predict(X_test)

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predicted_ratings, color='teal', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Movie Ratings')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.grid(True)
plt.tight_layout()
plt.show()




