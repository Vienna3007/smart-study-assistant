import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/dataset.csv")

X = data[['study_hours', 'completion_rate', 'score']]
y = data['weak_subject']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

def predict_weak(data):
    result = model.predict([data])[0]
    return "Weak Subject" if result == 1 else "Strong Subject"