import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle
import shap
import plotly.graph_objects as go
import json
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
st.title("Informations sur la décision d'accord du prêt au client")

# 1. Récupération de la liste des clients
try:
    clients_list = requests.get(
        "https://openclassrooms-datascientist-projet7.onrender.com/clients_list"
    ).json()
except requests.exceptions.RequestException as e:
    st.error(f"Erreur lors de la récupération des clients : {e}")
    st.stop()

# 2. Sélection du client
option = st.sidebar.selectbox(
    "Sélectionner l'identifiant du client",
    clients_list
)
st.sidebar.write(f"Client sélectionné : :blue[{option}]")

# 3. Récupération de la probabilité de défaut
params = {'id': option}
url_proba = "https://openclassrooms-datascientist-projet7.onrender.com/predict_proba"
try:
    response = requests.get(url_proba, params=params).json()
    predicted_failure_rate = response["predicted_failure_rate"]
    client_proba = np.round(predicted_failure_rate * 100, 1)
except requests.exceptions.RequestException as e:
    st.error(f"Erreur lors de la récupération de la probabilité : {e}")
    st.stop()
except KeyError:
    st.error("Erreur : La réponse de l'API ne contient pas 'predicted_failure_rate'.")
    st.stop()

# 4. Affichage de la décision et de la jauge
for_threshold = pd.read_csv("useful_saved_parameters.csv")
threshold = for_threshold["Model threshold"][0]

if client_proba > threshold:
    st.header(f"Prêt :red[refusé] - Probabilité estimée de défaut : {client_proba}%")
else:
    st.header(f"Prêt :green[accepté] - Probabilité estimée de défaut : {client_proba}%")

# Jauge
fig = go.Figure(go.Indicator(
    domain={'x': [0, 1], 'y': [0, 1]},
    value=client_proba,
    mode="gauge+number+delta",
    title={'text': f"Probabilité de faire défaut (seuil à {threshold}%)"},
    delta={
        'reference': threshold,
        'increasing': {'color': 'red'},
        'decreasing': {'color': 'green'}
    },
    gauge={
        'axis': {'range': [None, 100]},
        'steps': [
            {'range': [0, threshold], 'color': "green"},
            {'range': [threshold, 100], 'color': "red"}
        ],
        'threshold': {
            'line': {'color': "orange", 'width': 3},
            'thickness': 1,
            'value': threshold
        },
        'bar': {'color': 'darkblue'}
    }
))
st.plotly_chart(fig)

# 5. Explication de la décision (Waterfall plot)
st.header("1. Explication (optionnelle) de la prise de décision du modèle")

try:
    with open('explainer.sav', 'rb') as f:
        explainer = pickle.load(f)
    with open('feature_names.sav', 'rb') as f:
        feature_names = pickle.load(f)
except FileNotFoundError:
    st.error("Erreur : Le fichier explainer.sav ou feature_names.sav est introuvable.")
    st.stop()
except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers : {e}")
    st.stop()

# Récupération des features du client
url_features = "https://openclassrooms-datascientist-projet7.onrender.com/client_features_prep"
try:
    client_feats = requests.get(url_features, params=params).json()
except requests.exceptions.RequestException as e:
    st.error(f"Erreur lors de la récupération des features du client : {e}")
    st.stop()

# Waterfall plot
shap_values = explainer(np.array(client_feats))
shap_values.feature_names = feature_names

waterfall_box = st.checkbox(
    f'Afficher les paramètres les plus importants pour le client :blue[{option}]'
)
if waterfall_box:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    waterfall = shap.waterfall_plot(shap_values[0])
    st.pyplot(waterfall)

# 6. Affichage des caractéristiques du client
st.header("2. Affichage (optionnel) des caractéristiques du client")

# Récupération des features non scalées
url_client = "https://openclassrooms-datascientist-projet7.onrender.com/client_features"
try:
    my_client = requests.get(url_client, params=params).json()
except requests.exceptions.RequestException as e:
    st.error(f"Erreur lors de la récupération des données du client : {e}")
    st.stop()

data_client = pd.DataFrame()
data_client["feature_name"] = feature_names
data_client["feature_value"] = my_client[0]

options = st.multiselect(
    'Quelles caractéristiques du client afficher ?',
    data_client["feature_name"]
)
data_to_show = data_client.loc[data_client["feature_name"].isin(options), :]
st.write(data_to_show)

# 7. Comparaison avec des clients similaires
st.header("3. Comparaison avec des clients similaires (Même âge, mêmes revenus, même montant de crédit)")

# Récupération des features importantes
df = pd.DataFrame(shap_values[0].values)
df['feature_name'] = feature_names
df[0] = np.abs(df[0])
important_features = df.sort_values(0, ascending=False)["feature_name"][:9]
list_feats = important_features.values.tolist()

# Requête pour les clients similaires
data = {"id": option, "features": list_feats}
headers = {"Content-Type": "application/json; charset=utf-8"}
url = "https://openclassrooms-datascientist-projet7.onrender.com/similar_clients"
try:
    neighbors_features = requests.post(url, data=json.dumps(data), headers=headers)
    n_f = pd.DataFrame(neighbors_features.json())
except requests.exceptions.RequestException as e:
    st.error(f"Erreur lors de la récupération des clients similaires : {e}")
    st.stop()

feat_compare = st.multiselect(
    "Pour quelles features souhaitez-vous une comparaison ?",
    list_feats
)

for f in feat_compare:
    colors = ["green"] + ["blue"] * (len(n_f) - 1)
    fig, ax = plt.subplots()
    ax.bar(n_f.index, n_f[f], color=colors)
    plt.ylabel(f)
    plt.xlabel("Notre client (en vert) + 10 individus proches")
    plt.title(f"Valeurs de {f} pour les clients proches")
    st.pyplot(fig)

# 8. Informations bonus sur le client
age = data_client.loc[data_client['feature_name'] == "DAYS_BIRTH", "feature_value"].values[0] / (-365)
age = int(age)
st.sidebar.write(f"Âge : :blue[{age}] ans")

credit_amt = data_client.loc[data_client['feature_name'] == "AMT_CREDIT", "feature_value"].values[0]
credit_amt = int(credit_amt)
st.sidebar.write(f"Montant du crédit : :blue[{credit_amt}] $")

income = data_client.loc[data_client['feature_name'] == "AMT_INCOME_TOTAL", "feature_value"].values[0]
income = int(income)
st.sidebar.write(f"Revenu annuel : :blue[{income}] $")
