from __future__ import print_function
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from typing import Optional

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def main(path: Optional[str]=None):
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    flow = InstalledAppFlow.from_client_secrets_file(
        os.path.join(os.path.dirname(__file__),'credentials.json'), SCOPES)
    creds = flow.run_local_server(port=7075)
    # Save the credentials for the next run
    if path:
        with open(path, 'w') as token:
            token.write(creds.to_json())
    else:
        with open(os.path.join(os.path.dirname(__file__),'token.json'), 'w') as token:
            token.write(creds.to_json())

if __name__ == '__main__':
    main()