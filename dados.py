import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#dividir as colunas do csv que estão juntas
data = pd.read_csv('teste_indicium_precificacao.csv', sep =',', nrows = 5)

print(data.head())
print(data.dtypes)
print(data.describe())


#gráfico 
data.plot()
plt.show()