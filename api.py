# -*- coding: utf-8 -*-
#!/usr/bin/python
from flask import Flask, request
import joblib
import pandas as pd
import json

# Cargar el modelo entrenado
modelo = joblib.load('modelo_lr.pkl')

# Variables que usa el modelo
selected_features = [
    'acousticness',
    'danceability',
    'energy',
    'loudness',
    'instrumentalness',
]

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Recibir los parámetros de la URL (query parameters)
    acousticness = request.args.get('acousticness', type=float)
    danceability = request.args.get('danceability', type=float)
    energy = request.args.get('energy', type=float)
    loudness = request.args.get('loudness', type=float)
    instrumentalness = request.args.get('instrumentalness', type=float)
    
    # Verificar si todos los parámetros fueron recibidos
    if None in [acousticness, danceability, energy, loudness, instrumentalness]:
        return json.dumps({"error": "Faltan parámetros"}), 400, {'Content-Type': 'application/json'}
    
    # Crear un DataFrame con los valores recibidos
    X_new = pd.DataFrame({
        'acousticness': [acousticness],
        'danceability': [danceability],
        'energy': [energy],
        'loudness': [loudness],
        'instrumentalness': [instrumentalness],
    })
    
    # Hacer la predicción
    predicciones = modelo.predict(X_new)
    
    # Devolver las predicciones
    return json.dumps({'predicted_popularity': predicciones.tolist()}), 200, {'Content-Type': 'application/json'}

@app.route('/predict_Var2', methods=['GET'])
def predict_Var2():
    # Cargar datos de prueba
    dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2025/main/datasets/dataTest_Spotify.csv', index_col=0)
    
    # Seleccionar las características necesarias
    X_test = dataTesting[selected_features]
    
    # Tomar dos observaciones del conjunto de prueba para la validación
    validation_samples = X_test.iloc[:2].copy()
    
    # Hacer predicciones sobre estas observaciones
    predicciones = modelo.predict(validation_samples)
    
    # Preparar respuesta con las observaciones y sus predicciones
    resultados = []
    for i in range(len(validation_samples)):
        resultados.append({
            'observation': validation_samples.iloc[i].to_dict(),
            'predicted_popularity': float(predicciones[i])
        })
    
    return json.dumps({
        'validation_predictions': resultados,
        'message': 'Predicciones realizadas sobre 2 observaciones del conjunto de validación'
    }), 200, {'Content-Type': 'application/json'}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
