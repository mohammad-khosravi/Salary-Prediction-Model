import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess
df = pd.read_csv("data_jobs.csv")
df = df[df['salary_year_avg'].notnull()].copy()
df['job_skills'] = df['job_skills'].fillna('')
df['job_skills'] = df['job_skills'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) and x.startswith('[') else x)

# Sample
df_sample = df.sample(n=10000, random_state=42)

# Numeric conversion
df_sample['skills'] = df_sample['job_skills'].apply(lambda x: len(x.split()))
df_sample['title'] = df_sample['job_title_short'].astype('category').cat.codes
df_sample['location'] = df_sample['job_location'].astype('category').cat.codes
df_sample['schedule'] = df_sample['job_schedule_type'].astype('category').cat.codes
df_sample['country'] = df_sample['job_country'].astype('category').cat.codes

# Rename and select relevant columns
df_corr = df_sample.rename(columns={
    'salary_year_avg': 'salary',
    'job_work_from_home': 'remote',
    'job_no_degree_mention': 'no_degree',
    'job_health_insurance': 'insurance',
})[['salary', 'title', 'location', 'schedule', 'remote', 'no_degree', 'insurance', 'country', 'skills']]

# Compute correlation matrix
corr_matrix = df_corr.corr()

# Plot full 2D heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap='Reds', vmin=0, vmax=1, fmt=".2f", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()