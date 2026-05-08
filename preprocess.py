import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower().split()

    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]

    return " ".join(text)