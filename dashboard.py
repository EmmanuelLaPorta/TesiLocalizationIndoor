import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder


# Funzione caricamento validation dataset e modifiche
@st.cache_data 
def load_data():
    data = pd.read_csv('./Data/validationData.csv')
    data.drop(['SPACEID','RELATIVEPOSITION','USERID','PHONEID','TIMESTAMP'], axis=1, inplace=True )
    data['BUILDING_FLOOR'] = data['BUILDINGID']*10 + data['FLOOR']
    data.drop(['FLOOR','BUILDINGID'], axis=1, inplace=True)
    data = data.replace(to_replace=100,value=-111)
    #X_test = data[[i for i in data.columns if 'WAP' in i]]
    #y_test = data[[i for i in data.columns if not 'WAP' in i]]
    return data

# Funzione caricamento modelli basati su XGBoost
@st.cache_data()
def load_models():
    with open('modelli/xgboost_buildingfloor.pkl', "rb") as f1:
        buildingfloor_model = pickle.load(f1)
    with open('modelli/xgboost_latitude.pkl', "rb") as f2:
        latitude_model = pickle.load(f2)
    with open('modelli/xgboost_longitude.pkl', "rb") as f3:
        longitude_model = pickle.load(f3)

    return buildingfloor_model, latitude_model, longitude_model  

# Funzione predizione posizione 
def predict_positions(buildingfloor_model, latitude_model, longitude_model, input_data):

    buildingfloor_prediction_le = buildingfloor_model.best_estimator_.predict(input_data)
    #reverse encoding
    le = LabelEncoder()
    buildingfloor_prediction = le.inverse_transform(buildingfloor_prediction_le)

    latitude_prediction = latitude_model.best_estimator_.predict(input_data)

    longitude_prediction = longitude_model.best_estimator_.predict(input_data)
    
    return buildingfloor_prediction, latitude_prediction, longitude_prediction

# MAIN
def main():
    st.title("Dashboard di previsione della posizione utilizzando i modelli basati su XGBoost")
    
    # Carica i dati
    data = load_data()
    
    # Carica il modello
    buildingfloor_model, latitude_model, longitude_model = load_models()
    
    # Selettore di input per la riga del test set
    selected_row_index = st.sidebar.selectbox("Seleziona una riga dal test set (da 0 a 1110):", range(len(data)))
    selected_row = data.iloc[selected_row_index]
    st.sidebar.write("Dettagli della riga selezionata:")
    st.sidebar.write(selected_row)

    # Estrai l'edificio, la latitudine e la longitudine dalla riga selezionata
    buildingfloor_input = selected_row['BUILDING_FLOOR']
    lat_input = selected_row['LATITUDE']
    lon_input = selected_row['LONGITUDE']
        
    # Previsione della posizione
    input_data = np.array([[buildingfloor_input, lat_input, lon_input]])
    buildingfloor_prediction, latitude_prediction, longitude_prediction = predict_positions(buildingfloor_model, latitude_model, longitude_model, input_data)
    
    
    # Visualizzazione sulla mappa (plot dei dati)
    st.header("Posizione Predetta")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data['LATITUDE'], data['LONGITUDE'], hue=data['BUILDING_FLOOR'], palette='viridis', alpha=0.5)
    plt.scatter(latitude_prediction, longitude_prediction, color='red', label='Posizione Predetta')
    plt.xlabel('Latitudine')
    plt.ylabel('Longitudine')
    plt.title('Posizione Predetta sulla Mappa')
    plt.legend()
    st.pyplot()

    # Grafico a torta delle previsioni
    st.header("Proporzione delle Previsioni")
    prediction_counts = pd.Series(buildingfloor_prediction).value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Proporzione delle Previsioni')
    st.pyplot()

if __name__ == "__main__":
    main()