import numpy as np
import pandas as pd
import os # Importamos esto para arreglar las rutas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# --- TRUCO PARA ARREGLAR LA RUTA ---
# Esto busca la carpeta exacta donde está guardado ESTE archivo .py
directorio_actual = os.path.dirname(os.path.abspath(__file__))
# Esto une esa carpeta con el nombre del archivo
ruta_csv = os.path.join(directorio_actual, 'products.csv')

print(f"Buscando el archivo en: {ruta_csv}") # Esto te dirá dónde está buscando

# --- CARGAR DATOS ---
try:
    data = pd.read_csv(ruta_csv)
    print("¡ÉXITO! Datos cargados correctamente.")
except FileNotFoundError:
    print("\nERROR GRAVE: Sigue sin encontrar el archivo.")
    print("POR FAVOR REVISA: ")
    print("1. Que el archivo se llame 'products.csv' y no 'products.csv.txt'")
    print("2. Que el archivo esté AL LADO de este script de python.")
    exit()

# --- EL RESTO DEL CÓDIGO ---
features = data[['feature1', 'feature2', 'feature3']]
labels = data['label']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

def recommend(product_features):
    prediction = model.predict([product_features])
    return prediction[0]

# Prueba
print(f"Recomendación para [1.1, 2.1, 3.1]: {recommend([1.1, 2.1, 3.1])}")