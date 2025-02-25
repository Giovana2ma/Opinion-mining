import pandas as pd 
import numpy as np 
import re
import warnings
import nltk
import isodate
from nltk.corpus import stopwords
import unicodedata
from bertopic import BERTopic
warnings.filterwarnings('ignore')
import spacy
nlp_en = spacy.load("en_core_web_sm")
nlp_pt = spacy.load("pt_core_news_sm")
nlp_es = spacy.load("es_core_news_sm")


def read_data(path):
    data = pd.read_json(path, lines = True)
    return data

def filter_videos(data, terms):
    # Create a pattern to search for the terms
    pattern = r'\b(' + '|'.join(map(re.escape, terms)) + r')\b'

    # Filter rows where 'Title' or 'Description' contains any of the terms
    filtered_data = data[data['Title'].str.contains(pattern, case=False, na=False)]

    return filtered_data

def remove_tilde(text):
    if isinstance(text, str):  # Ensure it's a string
        return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    return text  # Return as is if not a string

# Function to remove URLs, numbers, and newlines
def remove_urls_numbers_newlines(text):
    text_without_urls = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text_without_numbers = re.sub(r'\d+', '', text_without_urls)  # Remove numbers
    text_without_newlines = re.sub(r'\n', ' ', text_without_numbers)  # Replace newlines with spaces
    return text_without_newlines

def get_stopwords():
    stop_words = set(stopwords.words('english')) | set(stopwords.words('portuguese')) | set(stopwords.words('spanish') )
    custom_stopwords = {"https", "follow","instagram","veja","abril","assine", \
                        "bitly","bit ly","abr","2vzw8dn","confira","últimas",\
                        "vejanoinsta", "br", "siga","axios","ina","fried",\
                        "summit","bbcnewsbrasil","samy","sf"}
    stop_words.update(custom_stopwords)
    stop_words.update(nlp_en.Defaults.stop_words)
    stop_words.update(nlp_pt.Defaults.stop_words)
    stop_words.update(nlp_es.Defaults.stop_words)

    return stop_words

def clean_text(text,lang):
    """Removes stopwords, punctuation, and special characters from text."""
    if pd.isna(text):  # Handle NaN values
        return ""

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters and punctuation
    text = remove_tilde(text)
    text = remove_urls_numbers_newlines(text)

    
    if lang == "en":
        doc = nlp_en(text)
    elif lang == "es":
        doc = nlp_es(text)
    else:
        doc = nlp_pt(text)

    stop_words = get_stopwords()

    words = [token.lemma_ for token in doc if token.lemma_ not in stop_words and not token.is_punct]
    return " ".join(words)

def get_topics(data):
    data['content'] = ' '.join(data['Title'],data['Description'])
    return 

def get_duration(data):
    data["Duration"] = data["Duration"].fillna("PT0M")
    data["Duration_minutes"] = data["Duration"].apply(lambda x: isodate.parse_duration(x).total_seconds()/60)
    return data["Duration_minutes"]


