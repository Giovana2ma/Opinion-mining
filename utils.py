import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
warnings.filterwarnings('ignore')


def read_data(path):
    data = pd.read_json(path, lines = True)
    return data

def barplot(data, y_cols,x_col,name):
    num_cols = len(y_cols)  # Número de colunas para plotar
    fig, axes = plt.subplots(num_cols,1, figsize=(5 * num_cols, 30))  # Criar subplots lado a lado
    
    if num_cols == 1:
        axes = [axes]  # Garantir que axes seja iterável mesmo com um gráfico só

    for ax, col in zip(axes, y_cols):
        data_sorted = data.sort_values(by=col, ascending=False)  # Ordenar pelo valor da coluna
        
        # Criar o gráfico de barras
        sns.barplot(x=x_col, y=col, data=data_sorted, palette='viridis', ax=ax)
        
        # Melhorar visualização
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Rotacionar rótulos
        ax.set_xlabel(name)
        ax.set_ylabel(col)
        ax.set_title(f'{col} by {name}')
        ax.set_yscale('log')  # Escala logarítmica para melhor distribuição dos valores

    plt.tight_layout()  # Ajustar layout para evitar sobreposição
    plt.show()

def filter_videos(data, terms):
    # Create a pattern to search for the terms
    pattern = r'\b(' + '|'.join(map(re.escape, terms)) + r')\b'

    # Filter rows where 'Title' or 'Description' contains any of the terms
    filtered_data = data[data['Title'].str.contains(pattern, case=False, na=False) |
                            data['Description'].str.contains(pattern, case=False, na=False)]

    return filtered_data  

def clean_text(text):
    """Removes stopwords, punctuation, and special characters from text."""

    stop_words = set(stopwords.words('english')) | set(stopwords.words('portuguese'))
    # Add custom stopwords
    custom_stopwords = {"https", "follow","instagram","veja","abril","assine" \
                        ,"bitly","bit ly","abr","2vzw8dn","confira","últimas"\
                          ,"vejanoinsta", "br", "siga" }
    stop_words.update(custom_stopwords)

    if pd.isna(text):  # Handle NaN values
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters and punctuation
    words = text.split()  # Tokenize text
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

def generate_wordcloud(data):
    """Generates a word cloud from the 'Title' and 'Description' columns."""
    # Merge 'Title' and 'Description' columns
    combined_text = " ".join(data["Title"].astype(str) + " " + data["Description"].astype(str))

    # Clean text
    cleaned_text = clean_text(combined_text)

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="viridis").generate(cleaned_text)

    # Display word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")  # Hide axes
    plt.title("Word Cloud - Title & Description", fontsize=14)
    plt.show()
