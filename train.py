import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("datasets/Fake.csv", encoding="latin1")
true = pd.read_csv("datasets/True.csv", encoding="latin1")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

# Use text column
X = data["text"]
y = data["label"]

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_vector = vectorizer.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))