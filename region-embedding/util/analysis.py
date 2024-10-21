import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leitura dos arquivos CSV
buildings_csv = pd.read_csv('./data/chicago-buildings.csv.gz')
pois_csv = pd.read_csv('./data/chicago-pois.csv.gz')
streets_complete_csv = pd.read_csv('./data/chicago-streets-complete.csv.gz')

# Leitura dos arquivos Parquet
buildings_parquet = pd.read_parquet('./data/chicago-buildings.parquet')
pois_parquet = pd.read_parquet('./data/chicago-pois.parquet')
streets_complete_parquet = pd.read_parquet('./data/chicago-streets-complete.parquet')

# Função para exibir informações básicas do DataFrame
def explore_data(df, name):
    print(f"\n{name} - Informações Básicas")
    print(df.info())
    print("\nPrimeiras linhas:")
    print(df.head())
    print("\nEstatísticas descritivas:")
    print(df.describe(include='all'))

# Função para calcular estatísticas descritivas detalhadas
def detailed_statistics(df, name):
    print(f"\n{name} - Estatísticas Descritivas Detalhadas")
    print("\nContagem de valores nulos:")
    print(df.isnull().sum())
    print("\nResumo estatístico:")
    print(df.describe(include='all'))

# Função para criar visualizações
def create_visualizations(df, name):
    print(f"\n{name} - Visualizações")
    
    # Histograma
    df.hist(bins=30, figsize=(15, 10))
    plt.suptitle(f'Histograma de {name}')
    plt.show()
    
    # Heatmap de correlação
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title(f'Heatmap de Correlação de {name}')
    plt.show()



# # Explorar os dados dos arquivos CSV
# explore_data(buildings_csv, "Buildings CSV")
# explore_data(pois_csv, "POIs CSV")
# explore_data(streets_complete_csv, "Streets Complete CSV")

# # Explorar os dados dos arquivos Parquet
# explore_data(buildings_parquet, "Buildings Parquet")
# explore_data(pois_parquet, "POIs Parquet")
# explore_data(streets_complete_parquet, "Streets Complete Parquet")

# # Estatísticas descritivas para os arquivos CSV
# detailed_statistics(buildings_csv, "Buildings CSV")
# detailed_statistics(pois_csv, "POIs CSV")
# detailed_statistics(streets_complete_csv, "Streets Complete CSV")

# # Estatísticas descritivas para os arquivos Parquet
# detailed_statistics(buildings_parquet, "Buildings Parquet")
# detailed_statistics(pois_parquet, "POIs Parquet")
# detailed_statistics(streets_complete_parquet, "Streets Complete Parquet")

# # Visualizações para os arquivos CSV
# create_visualizations(buildings_csv, "Buildings CSV")
# create_visualizations(pois_csv, "POIs CSV")
# create_visualizations(streets_complete_csv, "Streets Complete CSV")

# # Visualizações para os arquivos Parquet
# create_visualizations(buildings_parquet, "Buildings Parquet")
# create_visualizations(pois_parquet, "POIs Parquet")
# create_visualizations(streets_complete_parquet, "Streets Complete Parquet")
