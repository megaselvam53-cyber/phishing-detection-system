import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv("phishing.csv")

def extract_features(url):
    return [
        len(url),
        url.count('.'),
        1 if "https" in url else 0,
        1 if "@" in url else 0
    ]

X = data['url'].apply(extract_features).tolist()
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
print("Model trained successfully")
