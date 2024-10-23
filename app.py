import pandas as pd



# Carregar o arquivo Excel
arquivo_excel = 'VegaMetrics.xlsx'

# Ler a primeira planilha do arquivo Excel
df = pd.read_excel(arquivo_excel)

# Exibir as primeiras linhas da planilha


media_coluna = df.iloc[:, 8].mean()

print(f"A média da coluna 'Coluna1' é: {media_coluna}")




