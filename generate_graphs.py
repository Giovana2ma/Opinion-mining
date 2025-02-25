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
from process_text import *
warnings.filterwarnings('ignore')

def generate_month_year_barplot(data, name):
    data["PublicationDate"] = pd.to_datetime(data["PublicationDate"])

    data = data.dropna(subset=["PublicationDate"])
    data["upload_month_year"] = data["PublicationDate"].dt.to_period("M")
    count_data = data["upload_month_year"].value_counts().sort_index()

    # Configurar estilo do gráfico
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 5))

    # Criar o gráfico de barras
    sns.barplot(x=count_data.index.astype(str), y=count_data.values)

    # Configurar rótulos e título
    plt.xlabel("Month-Year", fontsize=12)
    plt.ylabel("Number of Videos", fontsize=12)
    plt.title(f"Distribution of Videos Over Time in {name}", fontsize=14, fontweight="bold")

    # Rotacionar os rótulos do eixo x para melhor legibilidade
    plt.xticks(rotation=45, ha="right")

    # Exibir o gráfico
    plt.show()

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


def generate_wordcloud(data,column,lang,name):
    combined_text = " ".join(data[column].astype(str))

    # Clean text
    cleaned_text = clean_text(combined_text,lang)

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="viridis").generate(cleaned_text)

    # Display word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")  # Hide axes
    plt.title(f"Word Cloud - {column} in {name}", fontsize=14)
    plt.show()

def generate_histogram(data,col,name):
    # Create the figure
    plt.figure(figsize=(8, 5))

    # Plot histogram with KDE
    sns.histplot(data[col], bins=10, kde=True, color="#4C72B0", edgecolor="black", alpha=0.8)

    # Labels and title
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Distribution of Video {col} in {name}", fontsize=14, fontweight="bold")

    # Grid for readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.grid(axis="x", linestyle="--", alpha=0.0)

    # Show plot
    plt.show()

def plot_term_histogram(data, terms,name):
    # Converter títulos para string (caso tenha valores NaN)
    data["Title"] = data["Title"].astype(str).str.lower()

    # Criar dicionário para contar ocorrências de cada termo
    term_counts = {
        term: data["Title"].str.contains(r'\b' + re.escape(term.lower()) + r'\b', regex=True).sum()
        for term in terms
    }

    # Criar DataFrame com os resultados e ordenar
    term_df = pd.DataFrame(term_counts.items(), columns=["Term", "Count"]).sort_values(by="Count", ascending=False)
    term_df = term_df[term_df['Count']>0]

    # Configurar estilo do gráfico
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    # Criar o gráfico de barras
    sns.barplot(y=term_df["Term"], x=term_df["Count"], palette="viridis")

    # Configurar rótulos e título
    plt.xlabel("Number of Appearances", fontsize=12)
    plt.ylabel("Term", fontsize=12)
    plt.title(f"Term Frequency in Video Titles in {name}", fontsize=14, fontweight="bold")

    # Exibir o gráfico
    plt.show()
