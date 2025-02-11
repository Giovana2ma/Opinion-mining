import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
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
