# CODESOFT-TASK2

🎬 IMDb Movie Rating Prediction

Predicting IMDb movie ratings using multiple regression models.

📁 Dataset:

Used IMDb Movies India.csv

Key features:

Name

Year

Duration

Genre

Director

Actors

Votes

Rating

🔧 Data Preprocessing:

Removed missing values in Rating

Extracted year from Name if Year was missing

Converted Duration to numeric (minutes)

Replaced missing text fields (Genre, Director, Actor 1-3) with "Unknown"

Cleaned and converted Votes column to integers

Extracted main genre from Genre

🔢 Feature Encoding:

Encoded categorical columns using LabelEncoder:

Director, Actor 1, Actor 2, Actor 3, Main Genre

Converted Year to float

🧠 Model Training & Prediction:

Selected features:

Year, Duration, Votes, Director, Actor 1, Actor 2, Actor 3, Main Genre

Target: Rating

Train-test split: 80% train, 20% test

Trained model:

Random Forest Regressor

Output:

Predicted ratings vs actual ratings (printed samples)

🤖 Model Comparison (RMSE):

Tested and evaluated the following models:

Linear Regression

Ridge Regression

Lasso Regression

Decision Tree

Random Forest

SVR (Support Vector Regression)

KNN (K-Nearest Neighbors)

Gradient Boosting

AdaBoost

XGBoost

Evaluation Metric: Root Mean Squared Error (RMSE)

Results printed for each model

📊 Visualization:

Plotted Actual vs Predicted Ratings

Graph shows how close predictions are to real values

Used matplotlib for plotting

🛠️ Tools Used:

Python

pandas

numpy

scikit-learn

xgboost

matplotlib
