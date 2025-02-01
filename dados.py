import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
import joblib

#1) Questão 

data = pd.read_csv("teste_indicium_precificacao.csv")

data.columns
print(data.count())
print(data.describe())
print(data['price'].median())
print(data['price'].mean())
data['bairro_group'].value_counts()
print(data.info())


data['bairro'] = data['bairro'].str.strip().str.lower()
preco_tipo = data.groupby('bairro_group')[['price']].mean().sort_values('price')
preco_tipo.plot(kind='barh', figsize=(14, 10), color='purple')
plt.title('Preço Médio por Bairro')
plt.xlabel('Preço Médio')
plt.ylabel('Bairro')
plt.show() 

plt.figure(figsize=(12, 6))
sns.scatterplot(x='minimo_noites', y='price', data=data)
plt.title('Preço vs. Mínimo de Noites')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='disponibilidade_365', y='price', data=data)
plt.title('Preço vs. Disponibilidade no Ano')
plt.show() 
data['nome'] = data['nome'].fillna('')

vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(data['nome'])
X = data.drop(['price', 'id', 'nome', 'host_id', 'host_name', 'ultima_review'], axis=1)
y = data['price']

X = pd.get_dummies(X, columns=['bairro_group', 'bairro', 'room_type'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


apartamento = {'id': 2595,
    'nome': 'Skylit Midtown Castle',
    'host_id': 2845,
    'host_name': 'Jennifer',
    'bairro_group': 'Manhattan',
    'bairro': 'Midtown',
    'latitude': 40.75362,
    'longitude': -73.98377,
    'room_type': 'Entire home/apt',
    'minimo_noites': 1,
    'numero_de_reviews': 45,
    'ultima_review': '2019-05-21',
    'reviews_por_mes': 0.38,
    'calculado_host_listings_count': 2,
    'disponibilidade_365': 355}

novo_df = pd.DataFrame([apartamento])
novo_df = novo_df.reindex(columns=X.columns, fill_value=0)


preco_previsto = model.predict(novo_df)
print(f"Preço previsto: ${preco_previsto[0]:.2f}")
#joblib.dump(model, 'modelo_precificacao.pkl')