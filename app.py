import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pickle
import pandas as pd
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input
from flask_cors import CORS
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

app = Flask(__name__)
CORS(app)

def normalize_class_name(class_name):
    return class_name.replace('_', ' ').replace('-', ' ').strip().lower()

# Fonction pour télécharger le modèle depuis Google Drive
def download_model_from_drive():
    SERVICE_ACCOUNT_FILE = './credentials.json'
    FILE_ID = '1R2D0VkO8E918X-SOoM2gAPotgkez1e_4'
    DESTINATION = 'plant_disease_efficientNetV2_modified2.h5'

    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    service = build('drive', 'v3', credentials=credentials)

    request = service.files().get_media(fileId=FILE_ID)
    fh = io.FileIO(DESTINATION, 'wb')
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")

    print(f"Le fichier a été téléchargé sous le nom {DESTINATION}")

# Télécharger le modèle
download_model_from_drive()

# Charger les noms de classes du fichier .pkl
with open('label_transform.pkl', 'rb') as f:
    lb = pickle.load(f)
    if isinstance(lb, (np.ndarray, list)):
        class_names = lb
    else:
        class_names = lb.classes_

# Normaliser les noms de classes
class_names_normalized = [normalize_class_name(name) for name in class_names]

# Charger le modèle
model = tf.keras.models.load_model('plant_disease_efficientNetV2_modified2.h5')

# Charger le fichier CSV
csv_file_path = './cleaned_plant_care.csv'
plant_care_df = pd.read_csv(csv_file_path, encoding='ISO-8859-1', delimiter=';')
plant_care_df['Normalized_Plante'] = plant_care_df['Plante'].apply(normalize_class_name)

# Créer le répertoire des uploads si nécessaire
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Prétraiter l'image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)[0]

    # Obtenir l'indice de la classe avec la plus haute probabilité
    top_pred_idx = np.argmax(preds)
    predicted_class = class_names[top_pred_idx].strip()
    predicted_class_normalized = normalize_class_name(predicted_class)

    # Trouver les informations de soins pour la classe prédite
    matched_rows = plant_care_df[plant_care_df['Normalized_Plante'] == predicted_class_normalized]
    if not matched_rows.empty:
        plant_info = matched_rows.iloc[0].to_dict()
        result = {
            "class": predicted_class,
            "probability": float(preds[top_pred_idx]),
            "plant_info": {
                "Plante": plant_info.get("Plante", "N/A"),
                "Maladie": plant_info.get("Maladie", "N/A"),
                "Description": plant_info.get("Description", "N/A"),
                "Symptômes": plant_info.get("Symptoms", "N/A"),  # Mise à jour
                "Traitement": plant_info.get("Traitements", "N/A"),  # Mise à jour
                "Prévention": plant_info.get("Prevention", "N/A")  # Mise à jour
            }
        }
    else:
        result = {
            "class": predicted_class,
            "probability": float(preds[top_pred_idx]),
            "plant_info": {
                "Plante": "N/A",
                "Maladie": "N/A",
                "Description": "N/A",
                "Symptômes": "N/A",
                "Traitement": "N/A",
                "Prévention": "N/A"
            }
        }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
