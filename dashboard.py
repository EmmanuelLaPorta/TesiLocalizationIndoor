import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Utilizza il backend 'Agg' per la visualizzazione non interattiva
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error

# Funzione caricamento dei dataset e modifiche
@st.cache_data 
def carica_dataset_test():
    #Caricamento del dataset.
    data = pd.read_csv('./Data/validationData.csv')
    #Drop colonne superflue
    data.drop(['SPACEID','RELATIVEPOSITION','USERID','PHONEID','TIMESTAMP'], axis=1, inplace=True )
    #Creazione BUILDING_FLOOR
    data['BUILDING_FLOOR'] = data['BUILDINGID']*10 + data['FLOOR']
    return data

@st.cache_data 
def carica_target_train():
    data = pd.read_csv('./Data/trainingData.csv')
    data['BUILDING_FLOOR'] = data['BUILDINGID']*10 + data['FLOOR']
    data = data[['BUILDING_FLOOR']]
    return data

# Funzione creazione features e modifiche
@st.cache_data 
def estrapola_features(data):
    X_test = data[[i for i in data.columns if 'WAP' in i]]
    X_test = X_test.replace(to_replace=100,value=-111)
    return X_test

# Funzione creazione targets e modifiche
@st.cache_data 
def estrapola_targets(data):
    y_test = data[[i for i in data.columns if not 'WAP' in i]]
    y_test.drop(['FLOOR','BUILDINGID'], axis=1, inplace=True)
    y_test = y_test.replace(to_replace=100,value=-111)
    return y_test

# Funzione caricamento modelli basati su XGBoost
@st.cache_data()
def carica_modelli():
    with open('modelli/xgboost_buildingfloor.pkl', "rb") as f1:
        buildingfloor_model = pickle.load(f1)
    with open('modelli/xgboost_latitude.pkl', "rb") as f2:
        latitude_model = pickle.load(f2)
    with open('modelli/xgboost_longitude.pkl', "rb") as f3:
        longitude_model = pickle.load(f3)

    return buildingfloor_model, latitude_model, longitude_model 

# Funzioni predizioni posizione

def predizione_encode(buildingfloor_model, X_test):
    #avvio encoding
    le = LabelEncoder()
    target = carica_target_train()
    le.fit_transform(target['BUILDING_FLOOR'])

    #predizione BUILDING_FLOOR
    predizione_encoded = buildingfloor_model.best_estimator_.predict(X_test)

    #reverse encoding
    predizione = le.inverse_transform(predizione_encoded)

    return predizione

def predizione(model, X_test):
    predizione = model.best_estimator_.predict(X_test)
    return predizione



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
import io

def grafico_predizione(data, real_bfl, real_lat, real_lon, pred_bfl, pred_lat, pred_lon):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    real_bld = real_bfl // 10
    real_flr = real_bfl % 10
    pred_flr = pred_bfl % 10

    for i in range(len(real_bld)):
        ax.scatter(data['LONGITUDE'], data['LATITUDE'], data['FLOOR'], alpha=0.5)

    # Grafico della posizione reale
    ax.scatter(real_lon, real_lat, real_flr, color='black', label='Posizione reale', s=100)

    # Grafico della posizione predetta
    ax.scatter(pred_lon, pred_lat, pred_flr, color='red', label='Posizione predetta', s=100)

    # Etichette degli assi
    ax.set_xlabel('Longitudine')
    ax.set_ylabel('Latitudine')
    ax.set_zlabel('Piano')

    # Aggiunta della legenda
    ax.legend()

    # Converti il grafico in un'immagine
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Visualizza l'immagine nel dashboard Streamlit
    st.image(img, caption='Predizione della Posizione', use_column_width=True, width=800)  # Ingrandisci l'immagine




# MAIN
def main():
    st.title("Dashboard di previsione della posizione con XGBoost")

    #carico i dati per il test
    data = carica_dataset_test()

    #carico i modelli
    buildingfloor, latitude, longitude = carica_modelli()

    #prendo le features per la predizione della riga
    feature = estrapola_features(data)

    #effettuo predizioni sulla riga selezionata
    pred_buildingfloor = predizione_encode(buildingfloor,feature)
    pred_latitude = predizione(latitude, feature)
    pred_longitude = predizione(longitude, feature)
    
    
    #seleziona posizione da visualizzare
    indice_riga = st.sidebar.selectbox("Seleziona una riga dal test set (da 0 a 1110):", range(len(data)))
    riga_selezionata = data.iloc[[indice_riga]] #voglio dataframe qui non series
    

    #mi conservo le posizioni reali
    real_riga_buildingfloor = riga_selezionata['BUILDING_FLOOR']
    real_riga_latitude = riga_selezionata['LATITUDE']
    real_riga_longitude = riga_selezionata['LONGITUDE']

    pred_riga_buildingfloor = pred_buildingfloor[indice_riga]
    pred_riga_latitude = pred_latitude[indice_riga]
    pred_riga_longitude = pred_longitude[indice_riga]

    #stampo su console le posizioni
    print("Edificio e piano reale:", real_riga_buildingfloor, "Edificio e piano predetti", pred_riga_buildingfloor)
    print("Latitudine reale:", real_riga_latitude, "Latitudine predetta", pred_riga_latitude)
    print("Longitudine reale:", real_riga_longitude, "Longitudine predetta", pred_riga_longitude)

    #stampo il grafico
    grafico_predizione(data, 
                       real_riga_buildingfloor,real_riga_latitude,real_riga_longitude,
                       pred_riga_buildingfloor,pred_riga_latitude,pred_riga_longitude)


    st.write("Dettagli della riga selezionata:")
    st.write(riga_selezionata)

    #stampo le metriche
    y_test = estrapola_targets(data)
    # Accuracy
    accuracy = accuracy_score(y_test['BUILDING_FLOOR'], pred_buildingfloor)
    st.write("Accuracy:", accuracy)

    # F1-score
    f1 = f1_score(y_test['BUILDING_FLOOR'], pred_buildingfloor, average='weighted')
    st.write("F1-score:", f1)

    # R2
    r2_latitude = r2_score(y_test['LATITUDE'], pred_latitude)
    st.write("R2 latitude:", r2_latitude)
    r2_longitude= r2_score(y_test['LONGITUDE'], pred_longitude)
    st.write("R2 longitude:", r2_longitude)

    # MSE
    mse_latitude = mean_squared_error(y_test['LATITUDE'], pred_latitude)
    st.write("MSE latitude:", mse_latitude)
    mse_longitude= mean_squared_error(y_test['LONGITUDE'], pred_longitude)
    st.write("MSE longitude:", mse_longitude)

    # MAE
    mae_latitude = mean_absolute_error(y_test['LATITUDE'], pred_latitude)
    st.write("MAE latitude:", mae_latitude)
    mae_longitude= mean_absolute_error(y_test['LONGITUDE'], pred_longitude)
    st.write("MAE longitude:", mae_longitude)
    



if __name__ == "__main__":
    main()

