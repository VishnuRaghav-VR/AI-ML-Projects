import joblib
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

model = MultinomialNB()
vectorizer = CountVectorizer()


df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1","v2"]]

X = df["v2"]  # Messages
y = df["v1"]  # Labels




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


X_train_numbers = vectorizer.fit_transform(X_train)
X_test_numbers = vectorizer.transform(X_test)

model.fit(X_train_numbers, y_train)


y_pred = model.predict(X_test_numbers)

accuracy = accuracy_score(y_test, y_pred)

# print("Accuracy:", accuracy)

# print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")