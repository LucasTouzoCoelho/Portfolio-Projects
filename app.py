from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

#load the trained model
model_and_scaler = joblib.load("final_churn_prediction.pkl")
model = model_and_scaler['model']
scaler = model_and_scaler['scaler']

label_encoders = joblib.load('label_encoders.pkl')  # Carregar o dicionário de encoders

training_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',  'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies',  'Contract',
     'PaperlessBilling', 'PaymentMethod','MonthlyCharges', 'TotalCharges'
]

@app.route("/")
def home():
    return "Churn Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    # Receber os dados enviados pelo usuário
    data = request.get_json(force=True)
    
    # Converter os dados para DataFrame
    df = pd.DataFrame([data])

    # Reorganizar as colunas para garantir que correspondem às colunas do treinamento
    try:
        df = df[training_columns]
    except KeyError as e:
        return jsonify({"error": f"Colunas ausentes nos dados: {str(e)}"}), 400
    
    # Converter as variáveis categóricas usando o LabelEncoder
    for column in label_encoders.keys():
        if column in df.columns and column not in ['SeniorCitizen','tenure', 'MonthlyCharges', 'TotalCharges']:
            # Use o LabelEncoder correspondente para transformar as categorias em números
            encoder = label_encoders[column]  # Carregar o encoder para a coluna
            df[column] = encoder.transform(df[column])

    try:
        # Aplicar o scaler nos dados
        input_data_scaled = scaler.transform(df)

        # Fazer a previsão
        prediction = model.predict(input_data_scaled)[0]
        probability = model.predict_proba(input_data_scaled)[0].tolist()

        return jsonify({"churn": int(prediction), "probability": probability})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)