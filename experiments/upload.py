import os
from boxsdk import Client, OAuth2
from tqdm import tqdm
import keyring


CLIENT_ID = os.environ["BOX_CLIENT_ID"]
CLIENT_SECRET = os.environ["BOX_CLIENT_SECRET"]


deepx_folder = "143536300530"
experiments_folder = "143809937071"
training_folder = "143811070013"


def save_tokens(access_token, refresh_token):
    print("Refreshing tokens...")


def load_tokens():
    access_token = keyring.get_password("access_token", "box")
    refresh_token = keyring.get_password("refresh_token", "box")
    return access_token, refresh_token


def get_client():
    oauth = OAuth2(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        store_tokens=save_tokens,
    )
    auth_url, csrf_token = oauth.get_authorization_url("https://epignatelli.com")
    auth_code = input("Application requires authorization. Visit {} and paste the part of the url after '&code':\n".format(auth_url))
    auth_code = str(auth_code)

    access_token, refresh_token = oauth.authenticate(auth_code)
    save_tokens(access_token, refresh_token)

    oauth = OAuth2(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        access_token=access_token,
        refresh_token=refresh_token,
    )

    client = Client(oauth)
    return client


def upload(filepath, client, dest_folder_id="0"):
    print("Uploading {} to folder {}".format(filepath, dest_folder_id))
    y = client.folder(dest_folder_id).upload(filepath)
    print("Upload successful")
    return y


if __name__ == "__main__":
    root = "/home/epignatelli/repos/cardiax/experiments/training/data/val"
    filenames = sorted(os.listdir(root))
    client = get_client()
    for i, filename in tqdm(enumerate(filenames)):
        try:
            print("Uploading file {}/{}".format(i + 1, len(filenames)))
            filepath = os.path.join(root, filename)
            upload(filepath, client, training_folder)
        except:
            continue
