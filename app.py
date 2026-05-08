from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# -----------------------------------
# Load Machine Learning Model
# -----------------------------------

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# -----------------------------------
# Load Dataset
# -----------------------------------

fake_news = pd.read_csv(
    "datasets/Fake.csv",
    encoding="latin1",
    engine="python"
)

true_news = pd.read_csv(
    "datasets/True.csv",
    encoding="latin1",
    engine="python"
)

news_data = pd.concat([fake_news, true_news])

# -----------------------------------
# Home Page
# -----------------------------------

@app.route("/")
def home():
    return render_template("home.html")

# -----------------------------------
# Fake News Detection
# -----------------------------------

@app.route("/detect", methods=["GET", "POST"])
def detect():

    prediction = None
    confidence = None
    related_news = None

    if request.method == "POST":

        news = request.form["news"]

        news = news.lower().strip()

        vector = vectorizer.transform([news])

        result = model.predict(vector)

        try:
            prob = model.predict_proba(vector)
            confidence = round(max(prob[0]) * 100, 2)
        except:
            confidence = 90

        if result[0] == 0:
            prediction = "⚠ Fake News Detected"
        else:
            prediction = "✅ Real News"

        related_news = news_data.sample(5)["text"].tolist()

    return render_template(
        "detect.html",
        prediction=prediction,
        confidence=confidence,
        related_news=related_news
    )

# -----------------------------------
# Dataset Viewer
# -----------------------------------

@app.route("/news")
def news():

    sample_news = news_data.sample(20)

    table = sample_news.to_html(classes="table table-striped")

    return render_template("news.html", table=table)

# -----------------------------------
# Countries List
# -----------------------------------

@app.route("/countries")
def countries():

    countries = [
        "India",
        "USA",
        "United Kingdom",
        "Canada",
        "Australia",
        "Germany",
        "France",
        "Japan",
        "China",
        "Brazil",
        "Russia",
        "Italy",
        "Spain",
        "South Korea",
        "Singapore",
        "Netherlands",
        "UAE",
        "South Africa",
        "Indonesia",
        "Mexico"
    ]

    return render_template("countries.html", countries=countries)

# -----------------------------------
# Country Categories
# -----------------------------------

@app.route("/country/<name>")
def country(name):

    categories = [
        "Politics",
        "Sports",
        "Technology",
        "Business",
        "Health",
        "Entertainment"
    ]

    return render_template(
        "categories.html",
        country=name,
        categories=categories
    )

# -----------------------------------
# Category News
# -----------------------------------

import requests
import random

API_KEY = "4e3a501a173e462597062beece3a055a"

@app.route("/country/<name>/<category>")
def category_news(name, category):

    # create search query
    query = f"{name} {category}"

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&"
        f"language=en&"
        f"pageSize=32&"
        f"apiKey={API_KEY}"
    )

    response = requests.get(url).json()

    articles = []

    if "articles" in response:

        random.shuffle(response["articles"])

        for a in response["articles"][:32]:

            articles.append({
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
                "image": a.get("urlToImage")
            })

    return render_template(
        "category_news.html",
        country=name,
        category=category,
        articles=articles
    )
# -----------------------------------
# Analytics Page (ADDED)
# -----------------------------------

@app.route("/analytics")
def analytics():

    fake_count = len(fake_news)
    true_count = len(true_news)

    total = fake_count + true_count

    fake_percent = round((fake_count / total) * 100, 2)
    true_percent = round((true_count / total) * 100, 2)

    return render_template(
        "analytics.html",
        fake_count=fake_count,
        true_count=true_count,
        fake_percent=fake_percent,
        true_percent=true_percent

    )

# -----------------------------------
# Explain Model
# -----------------------------------

@app.route("/explain")
def explain():

    explanation = [
        "News text is cleaned and processed.",
        "Text is converted into numbers using vectorization.",
        "Machine learning model analyzes patterns.",
        "System predicts whether the news is fake or real."
    ]

    return render_template(
        "explain.html",
        explanation=explanation
    )

# -----------------------------------
# Run Flask App
# -----------------------------------

if __name__ == "__main__":
    app.run(debug=True)