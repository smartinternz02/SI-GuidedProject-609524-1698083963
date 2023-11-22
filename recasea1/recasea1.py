#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the text file
data_txt = 'household_power_consumption.txt'
data = pd.read_csv(data_txt, delimiter=';')

# Save as CSV file
data.to_csv('household_power_consumption.csv', index=False)

data.dropna(inplace=True)  # Drop rows with missing values
data.isnull().sum()

def preprocess(data):
    # Convert date and time columns to datetime format
    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data.drop(['Date', 'Time'], axis=1, inplace=True)  # Drop original date and time columns

    # Extract features from date and time
    data['hour'] = data['datetime'].dt.hour
    data['period'] = pd.cut(data['hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])

    label_encoder = LabelEncoder()
    data['period_e'] = label_encoder.fit_transform(data['period'])

    # Displaying the mapping of encoded labels to original values
    label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print("Encoded Labels and Original Values Mapping:")
    print(label_mapping)

    # Display the encoded data with the 'period_encoded' column
    print(data.head())

    # Drop the original 'period' column if needed
    data.drop('period', axis=1, inplace=True)

    numeric_columns = ['Global_reactive_power', 'Global_intensity', 'Voltage']
    data[numeric_columns] = data[numeric_columns].astype('float64')

    data['is_weekend'] = data['datetime'].dt.dayofweek // 5  # Binary flag for weekends
    # Feature Engineering - Appliance Contribution Analysis
    new_column_names = {
        'Sub_metering_1': 'kitchen_cons',
        'Sub_metering_2': 'laundry_cons',
        'Sub_metering_3': 'heatcool_cons'
    }

    data = data.rename(columns=new_column_names)

    data['Global_active_power'] = pd.to_numeric(data['Global_active_power'], errors='coerce')
    data['kitchen_cons'] = pd.to_numeric(data['kitchen_cons'], errors='coerce')
    data['laundry_cons'] = pd.to_numeric(data['laundry_cons'], errors='coerce')
    data['heatcool_cons'] = pd.to_numeric(data['heatcool_cons'], errors='coerce')

    # Calculate the total active energy consumed every minute by electrical equipment not measured in sub-meterings 1, 2, and 3
    data['other_cons'] = (data['Global_active_power'] * 1000 / 60) - data['kitchen_cons'] - data['laundry_cons'] - data['heatcool_cons']

    # Set 'datetime' as the index
    data.set_index('datetime', inplace=True)
    data.head(10)

    data.to_csv('processed_data.csv', index=False)
    return data

data = preprocess(data)

'''
# Aggregate appliance consumption over a specific time period (e.g., per day)
appliance_consumption = data[['datetime', 'kitchen_cons', 'laundry_cons', 'heatcool_cons', 'other_cons']].groupby('datetime').sum()
print(appliance_consumption)

# Visualization - Appliance Contribution Pie Chart
plt.figure(figsize=(8, 6))
labels = ['Kitchen', 'Laundry', 'Water Heater & AC', 'Other']
plt.pie(appliance_consumption.mean(), labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Average Daily Appliance Contribution')
plt.axis('equal')
plt.show()
'''

# Histogram of Global Active Power
plt.figure(figsize=(10, 6))
sns.histplot(data['Global_active_power'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Global Active Power')
plt.xlabel('Global Active Power (kW)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Global_reactive_power'], bins=30, kde=True, color='darkblue')
plt.title('Distribution of Global Reactive Power')
plt.xlabel('Global REActive Power (kW)')
plt.ylabel('Frequency')
plt.show()

# Compute correlation matrix
correlation_matrix = data.corr(numeric_only=True)

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Features and target variable
features = data[['Global_reactive_power', 'Voltage', 'Global_intensity', 'kitchen_cons', 'laundry_cons', 'heatcool_cons', 'period_e', 'is_weekend', 'other_cons']]
target = data['Global_active_power']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=80, max_depth=10, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

filename= 'recaseamodel.pkl'
pickle.dump(model, open(filename,'wb'))