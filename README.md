PROJECT NOTES:

# Company name: `ConnectSphere - Degital AD`

Issue of company : "ConnectSphere Digital is wasting its clients' money by showing ads to everyone, instead of focusing on the people 
                    who are actually likely to be Interested. They need a smarter, Data-driven way to find the right audience.
                    
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
try:
    ad_data = pd.read_csv('advertising.csv')
except FileNotFoundError:
    print("Error: 'advertising.csv' not found. Please ensure the file is in the same directory.")
    exit()

# --- Exploratory Data Analysis ---

print("--- Data Info ---")
ad_data.info()
print("\n--- Descriptive Statistics ---")
print(ad_data.describe())

# Check for missing values
print("\n--- Missing Values ---")
print(ad_data.isnull().sum())


# Visualizing the data
sns.set_style('whitegrid')

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(ad_data['Age'], bins=30, kde=True)
plt.title('Age Distribution of Users')
plt.xlabel('Age')
plt.show()

# Jointplot of Area Income vs. Age
plt.figure(figsize=(10, 6))
sns.jointplot(x='Age', y='Area Income', data=ad_data)
plt.suptitle('Area Income vs. Age', y=1.02)
plt.show()

# Jointplot of Daily Time Spent on Site vs. Age
plt.figure(figsize=(10, 6))
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind='kde', color='red')
plt.suptitle('Daily Time Spent on Site vs. Age (Kernel Density)', y=1.02)
plt.show()

# Jointplot of Daily Time Spent on Site vs. Daily Internet Usage
plt.figure(figsize=(10, 6))
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data, color='green')
plt.suptitle('Daily Time Spent on Site vs. Daily Internet Usage', y=1.02)
plt.show()

# Pairplot to visualize relationships between numerical features, colored by 'Clicked on Ad'
sns.pairplot(ad_data, hue='Clicked on Ad', vars=['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage'], palette='viridis')
plt.suptitle('Pairwise Relationships of Key Features by Ad Click Outcome', y=1.02)
plt.show()


# --- Logistic Regression Model ---

# Define features (X) and target (y)
features = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']
X = ad_data[features]
y = ad_data['Clicked on Ad']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Create and train the logistic regression model
logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(X_train, y_train)

# --- Model Evaluation ---

# Make predictions on the test set
predictions = logmodel.predict(X_test)

# Print the classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, predictions))

# Generate and visualize the confusion matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, predictions)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Clicked', 'Clicked'], yticklabels=['Not Clicked', 'Clicked'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
```
