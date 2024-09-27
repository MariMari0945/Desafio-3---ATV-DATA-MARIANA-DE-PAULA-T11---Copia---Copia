import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importar os dados do CSV
df = pd.read_csv('data/medical_examination.csv')

# Criar a coluna 'overweight' (acima do peso)
df['overweight'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = df['overweight'].apply(lambda x: 1 if x > 25 else 0)

# Normalizar os dados de colesterol e glicose
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

def cat_plot():
    # Criar DataFrame para o gráfico categórico usando pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Desenhar o gráfico categórico com seaborn sem agrupar manualmente
    fig = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='count').fig

    return fig

def heat_map():
    # Limpar os dados
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # Calcular a matriz de correlação
    corr = df_heat.corr()

    # Gerar uma máscara para o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Configurar a figura do matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # Desenhar o heatmap (mapa de calor)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", center=0, square=True, cmap='coolwarm', linewidths=0.5)

if __name__ == "__main__":
    
    # Plota o gráfico categórico 
    fig_cat = cat_plot()
    plt.show()  

    # Plota o gráfico de calor
    fig_heat = heat_map()
    plt.show()
