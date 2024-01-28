
import pandas as pd
from google.colab import files
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Function to read and preprocess regular season stats
def preprocess_regular_season_stats(file_path):
    df = pd.read_csv(file_path)
    df[['Team', 'Statistic', 'Value']] = df['Team,Statistic,Value'].str.rsplit(',', 2, expand=True)
    df.drop('Team,Statistic,Value', axis=1, inplace=True)
    return df

# Function to read and preprocess head-to-head stats
def preprocess_head_to_head(file_path):
    df = pd.read_csv(file_path)
    df[['Date', 'Result', 'Location']] = df['Date,Result,Location'].str.split(',', 2, expand=True)
    df.drop('Date,Result,Location', axis=1, inplace=True)
    return df

# Upload and preprocess each dataset
print("Upload the Regular Season Stats CSV file:")
uploaded = files.upload()
regular_season_stats = preprocess_regular_season_stats(next(iter(uploaded.keys())))

print("\nUpload the Weather CSV file:")
uploaded = files.upload()
weather_data = pd.read_csv(next(iter(uploaded.keys())))  # No preprocessing needed yet

print("\nUpload the 49ers vs Lions Head-to-Head Stats CSV file:")
uploaded = files.upload()
head_to_head_49ers_lions = preprocess_head_to_head(next(iter(uploaded.keys())))

# Feature engineering (similar to your existing code)

# Combine features into a single DataFrame
combined_features = regular_season_stats
combined_features['HeadToHeadWins'] = head_to_head_wins

# Create 'Target' column based on 'HeadToHeadWins'
combined_features['Target'] = combined_features['HeadToHeadWins'].apply(lambda wins: 1 if wins > 10 else 0)

# Drop unnecessary columns
combined_features.drop(['HeadToHeadWins'], axis=1, inplace=True)

# Extract numeric columns for imputation
numeric_columns = combined_features.select_dtypes(include=['number']).columns
X_numeric = combined_features[numeric_columns]

# Impute missing values only for numeric columns
imputer = SimpleImputer(strategy='mean')
X_numeric_imputed = imputer.fit_transform(X_numeric)

# Create a DataFrame with imputed values
X_imputed = pd.DataFrame(X_numeric_imputed, columns=X_numeric.columns)

# Check the updated DataFrame
print(X_imputed.head())

# Specify the target variable
y_teams = combined_features['Target']

# Initialize the Random Forest Classifier model
model = RandomForestClassifier()

# Train the model
model.fit(X_imputed, y_teams)

# Make predictions for the specified teams
predictions = model.predict(X_imputed)

teams_to_predict = ['Lions', '49ers']
prediction_team_1 = predictions[0]
opposing_team_1 = teams_to_predict[1] if teams_to_predict[0] == 'Lions' else 'Lions'

if prediction_team_1:
    print(f"Prediction for {teams_to_predict[0]}: Win")
    print(f"Prediction for {opposing_team_1}: Loss")
else:
    print(f"Prediction for {teams_to_predict[0]}: Loss")
    print(f"Prediction for {opposing_team_1}: Win")
