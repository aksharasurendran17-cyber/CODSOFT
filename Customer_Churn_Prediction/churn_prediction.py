import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("churn.csv")

# remove customerID column if present
if 'customerID' in data.columns:
    data = data.drop('customerID', axis=1)

# convert target column to numeric
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# convert categorical features to numeric
data = pd.get_dummies(data)

# features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestClassifier()

# train model
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))