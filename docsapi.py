""" Google Drive API tools """

# from __future__ import print_function
import pickle
import os.path
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


def build_service():
    """ builds google drive service """
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = [
        # 'https://www.googleapis.com/auth/drive.metadata.readonly',
        'https://www.googleapis.com/auth/drive'
    ]
    
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    return service


def get_file(service, file_id):
    """ """
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        _, done = downloader.next_chunk()
        # print(f"Download {int(status.progress() * 100)}")
    return fh


def find_files(service, lbl):
    """ """
    return service.files().list(
        q=f"name contains '{lbl}'",
        spaces='drive',
        fields='nextPageToken, files(id, name)'
    ).execute().get('files', [])

    # return response.get('files', [])

def run_sync(service, lbl, path):
    items = find_files(service, lbl)
    if not items:
        print('Items not found')
        return

    for x in items:
        content = get_file(service, x['id'])
        print(x['name'])
        with open(os.path.join(path, x['name']), 'wb') as f:
            f.write(content.getvalue())


def sync_models(service):
    run_sync(service, 'nn_model_3d', './dat')


def sync_norm(service):
    run_sync(service, 'norm_', './dat')


def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    service = build_service()
    sync_models(service)
    sync_norm(service)


if __name__ == '__main__':
    main()
