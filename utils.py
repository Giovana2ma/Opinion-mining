import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from wordcloud import WordCloud
import nltk
import isodate
from nltk.corpus import stopwords
import unicodedata
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

def clean_text(text):
    """Removes stopwords, punctuation, and special characters from text."""

    stop_words = set(stopwords.words('english')) | set(stopwords.words('portuguese'))
    # Add custom stopwords
    custom_stopwords = {"https", "follow","instagram","veja","abril","assine" \
                        ,"bitly","bit ly","abr","2vzw8dn","confira","últimas"\
                          ,"vejanoinsta", "br", "siga"}
    stop_words.update(custom_stopwords)

    if pd.isna(text):  # Handle NaN values
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters and punctuation
    text = remove_tilde(text)
    text = remove_urls_numbers_newlines(text)
    words = text.split()  # Tokenize text
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

def generate_wordcloud(data):
    """Generates a word cloud from the 'Title' and 'Description' columns."""
    # Merge 'Title' and 'Description' columns
    combined_text = " ".join(data["Title"].astype(str))

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

def generate_duration_histogram(data,name):
        data["Duration_minutes"] = data["Duration"].apply(lambda x: isodate.parse_duration(x).total_seconds()/60)
        # sns.set_style("whitegrid")

        # Create the figure
        plt.figure(figsize=(8, 5))

        # Plot histogram with KDE
        sns.histplot(data["Duration_minutes"], bins=10, kde=True, color="#4C72B0", edgecolor="black", alpha=0.8)

        # Labels and title
        plt.xlabel("Duration (minutes)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"Distribution of Video Durations on {name}", fontsize=14, fontweight="bold")

        # Grid for readability
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.grid(axis="x", linestyle="--", alpha=0.0)

        # Show plot
        plt.show()
def generate_histogram(data,col,name):
        # Create the figure
        plt.figure(figsize=(8, 5))

        # Plot histogram with KDE
        sns.histplot(data[col], bins=10, kde=True, color="#4C72B0", edgecolor="black", alpha=0.8)

        # Labels and title
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"Distribution of Video {col} on {name}", fontsize=14, fontweight="bold")

        # Grid for readability
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.grid(axis="x", linestyle="--", alpha=0.0)

        # Show plot
        plt.show()

def generate_month_year_barplot(data, name):
    data["PublicationDate"] = pd.to_datetime(data["PublicationDate"])

    data = data.dropna(subset=["PublicationDate"])
    data["upload_month_year"] = data["PublicationDate"].dt.to_period("M")
    count_data = data["upload_month_year"].value_counts().sort_index()

    # Configurar estilo do gráfico
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 5))

    # Criar o gráfico de barras
    sns.barplot(x=count_data.index.astype(str), y=count_data.values, palette="viridis")

    # Configurar rótulos e título
    plt.xlabel("Month-Year", fontsize=12)
    plt.ylabel("Number of Videos", fontsize=12)
    plt.title(f"Distribution of Videos Over Time on {name}", fontsize=14, fontweight="bold")

    # Rotacionar os rótulos do eixo x para melhor legibilidade
    plt.xticks(rotation=45, ha="right")

    # Exibir o gráfico
    plt.show()
