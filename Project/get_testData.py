import pandas as pd
import numpy as np
import requests 
import json 

if __name__ == '__main__':
    f = open('credentials.json',)
    creds = json.load(f)
    CLIENT_ID = creds["client_id"]
    CLIENT_SECRET = creds["client_secret"]
    f.close()

    body_params = {"grant_type" : "client_credentials"}

    url='https://accounts.spotify.com/api/token'

    response = requests.post(url, data=body_params, auth = (CLIENT_ID, CLIENT_SECRET))

    token = response.json()["access_token"]

    headers = {'Authorization': 'Bearer '+ token}

    # get classical songs + extract track id's 
    r = requests.get("https://api.spotify.com/v1/playlists/37i9dQZF1DX48TTZL62Yht/tracks", headers=headers)
    c_songs = r.json()
    c_ids = []
    for song in c_songs["items"]:
        c_ids.append(song['track']['id'])

    c_ids = ",".join(c_ids)

    r = requests.get(f"https://api.spotify.com/v1/audio-features/?ids={c_ids}", headers=headers)
    features = r.json()

    df_c = pd.DataFrame(features["audio_features"])
    df_c["genre"] = "Hip-Hop"
    df_c.to_csv(r'data/rock_test.csv', index = False)