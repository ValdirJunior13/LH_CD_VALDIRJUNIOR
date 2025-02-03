## Descrição do Projeto

Este projeto tem como objetivo desenvolver um modelo de Machine Learning para prever os preços de aluguéis temporários em Nova York. Utilizando um conjunto de dados contendo informações sobre propriedades listadas para aluguel, aplicamos análise exploratória de dados (EDA) e um modelo de regressão Random Forest para realizar as previsões.

1. Relatório e Resultados

O relatório da análise exploratória e dos resultados do modelo pode ser encontrado no arquivo relatorio.pdf. As principais descobertas e insights sobre os fatores que influenciam o preço estão descritos no relat

2. Modelo Utilizado
O modelo escolhido foi um Random Forest Regressor, pois ele tem bom desempenho para previsão de valores numéricos e é robusto contra outliers e features irrelevantes. Ele foi avaliado utilizando a métrica R² (coeficiente de determinação) e MAE (Mean Absolute Error) para medir sua precisão.

## Instalação

1. Clone o repositório
    git clone https://github.com/ValdirJunior13/LH_CD_VALDIRJUNIOR 

2. Instalando as dependências
    pip install -r requirements.txt

3. Executando o Projeto 
Processamento dos Dados: 
Execute o script dados.py para processar os dados:
    python dados.py

4. Análise e Modelagem
Abra o notebook dados.ipynb para visualizar a análise dos dados e o processo de modelagem:
    jupyter notebook dados.ipynb
