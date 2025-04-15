import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv("data_jobs.csv")
df = df[df['salary_year_avg'].notnull()].copy()

# Clean and process job_skills
df['job_skills'] = df['job_skills'].fillna('')
df['job_skills'] = df['job_skills'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) and x.startswith('[') else x)

# Feature selection
features = ['job_title_short', 'job_location', 'job_schedule_type', 'job_work_from_home',
            'job_no_degree_mention', 'job_health_insurance', 'job_country', 'job_skills']
target = 'salary_year_avg'

# Sample for efficiency
df_sample = df.sample(n=10000, random_state=42)
X = df_sample[features]
y = df_sample[target]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('title', OneHotEncoder(handle_unknown='ignore', max_categories=20), ['job_title_short']),
    ('location', OneHotEncoder(handle_unknown='ignore', max_categories=20), ['job_location']),
    ('schedule', OneHotEncoder(handle_unknown='ignore'), ['job_schedule_type']),
    ('country', OneHotEncoder(handle_unknown='ignore', max_categories=20), ['job_country']),
    ('skills', CountVectorizer(max_features=100), 'job_skills'),
], remainder='passthrough')

# Pipeline with Linear Regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression RMSE: {rmse:.2f}")
print(f"Linear Regression R²: {r2:.4f}")

# Plot predictions
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.3, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Linear Regression: Actual vs Predicted Salary")
plt.grid(True)
plt.tight_layout()
plt.show()