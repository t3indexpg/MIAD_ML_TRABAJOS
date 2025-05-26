# -*- coding: utf-8 -*-
#!/usr/bin/python
from flask import Flask, request
import joblib
import pandas as pd
import json

# Cargar el modelo entrenado
modelo_ = joblib.load('modelo_logistic_regr.pkl')
modelo = modelo_['modelo']
vectorizer = modelo_['vectorizer']


app = Flask(__name__)


@app.route('/predict_Var2', methods=['GET'])
def predict_Var2():
    # Cargar datos de prueba
    dataTesting = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)

    # Tomar observaciones del conjunto de prueba
    plots = dataTesting['plot'].iloc[:5].copy()
    
    # Vectorizar los textos de prueba
    X_test_vectorized = vectorizer.transform(plots)
    
    # Hacer predicciones sobre las observaciones vectorizadas
    predicciones = modelo.predict(X_test_vectorized)
    
    # Preparar respuesta
    resultados = []
    for i in range(len(plots)):
        resultados.append({
            'plot': plots.iloc[i],
            'predicted_genres': predicciones[i].tolist()  
        })

    return json.dumps({
        'validation_predictions': resultados,
        'message': 'Predicciones realizadas sobre 5 observaciones del conjunto de validación'
    }), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
