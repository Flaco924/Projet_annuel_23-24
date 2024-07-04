from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Le chemin vers votre fichier de clé JSON
SERVICE_ACCOUNT_FILE = 'credentials.json'

# ID du fichier du modèle sur Google Drive (à obtenir depuis l'URL de partage du fichier)
FILE_ID = '1R2D0VkO8E918X-SOoM2gAPotgkez1e_4'
DESTINATION = 'plant_disease_model.h5'

# Autorisations requises
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Créer un service Google Drive API
service = build('drive', 'v3', credentials=credentials)

# Télécharger le fichier depuis Google Drive
request = service.files().get_media(fileId=FILE_ID)
fh = io.FileIO(DESTINATION, 'wb')
downloader = MediaIoBaseDownload(fh, request)

done = False
while done is False:
    status, done = downloader.next_chunk()
    print("Download %d%%." % int(status.progress() * 100))

print(f"Le fichier a été téléchargé sous le nom {DESTINATION}")
