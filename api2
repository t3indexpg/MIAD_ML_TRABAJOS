# -*- coding: utf-8 -*-
#!/usr/bin/python
from flask import Flask, request
import joblib
import pandas as pd
import json

# Cargar el modelo entrenado
modelo = joblib.load('modelo_logistic_regr.pkl')


app = Flask(__name__)


@app.route('/predict_Var2', methods=['GET'])
def predict_Var2():
    # Cargar datos de prueba
    dataTesting = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)
  
    
    # Tomar dos observaciones del conjunto de prueba para la validación
    validation_samples = X_test.iloc[:5].copy()
    
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
