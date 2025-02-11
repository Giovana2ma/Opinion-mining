import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def read_data(path):
    data = pd.read_json(path, lines = True)
    return data

def barplot(data, cols):
    num_cols = len(cols)  # Número de colunas para plotar
    fig, axes = plt.subplots(1, num_cols, figsize=(10 * num_cols, 10))  # Criar subplots lado a lado
    
    if num_cols == 1:
        axes = [axes]  # Garantir que axes seja iterável mesmo com um gráfico só

    for ax, col in zip(axes, cols):
        data_sorted = data.sort_values(by=col, ascending=False)  # Ordenar pelo valor da coluna
        
        # Criar o gráfico de barras
        sns.barplot(x='Title', y=col, data=data_sorted, palette='viridis', ax=ax)
        
        # Melhorar visualização
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Rotacionar rótulos
        ax.set_xlabel('Channel')
        ax.set_ylabel(col)
        ax.set_title(f'{col} by Channel')
        ax.set_yscale('log')  # Escala logarítmica para melhor distribuição dos valores

    plt.tight_layout()  # Ajustar layout para evitar sobreposição
    plt.show()
