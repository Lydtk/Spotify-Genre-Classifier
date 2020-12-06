import pandas as pd
import numpy as np
import requests 
import json 

def shuffle_data():
    df = pd.read_csv("data/song_data_train.csv")
    shuffled_df = df.sample(frac=1)
    shuffled_df.to_csv("data/train_shuffled.csv", index=False)

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

    # [hip-hop, classical, rock]    
    playlists = ["37i9dQZF1DX48TTZL62Yht", "37i9dQZF1DWWEJlAGA9gs0", "37i9dQZF1DWXRqgorJj26U"]

    # get hip-hop songs + extract track id's 
    r = requests.get("https://api.spotify.com/v1/playlists/37i9dQZF1DX48TTZL62Yht/tracks", headers=headers)
    hh_songs = r.json()
    hh_ids = []
    for song in hh_songs["items"]:
        hh_ids.append(song['track']['id'])

    # get classical songs + extract track id's 
    r = requests.get("https://api.spotify.com/v1/playlists/37i9dQZF1DWWEJlAGA9gs0/tracks", headers=headers)
    c_songs = r.json()
    c_ids = []
    for song in c_songs["items"]:
        c_ids.append(song['track']['id'])

    # get rock songs + extract track id's 
    r = requests.get("https://api.spotify.com/v1/playlists/37i9dQZF1DWXRqgorJj26U/tracks", headers=headers)
    r_songs = r.json()
    r_ids = []
    for song in r_songs["items"]:
        r_ids.append(song['track']['id'])

    ## get features for individual tracks, leverage spotify multi-track request
    hh_ids = ",".join(hh_ids)
    c_ids = ",".join(c_ids)
    r_ids = ",".join(r_ids)
    
    r = requests.get(f"https://api.spotify.com/v1/audio-features/?ids={hh_ids}", headers=headers)
    features = r.json()

    df_hh = pd.DataFrame(features["audio_features"])
    df_hh["genre"] = "Hip-Hop"

    r = requests.get(f"https://api.spotify.com/v1/audio-features/?ids={c_ids}", headers=headers)
    features = r.json()

    df_c = pd.DataFrame(features["audio_features"])
    df_c["genre"] = "Classical"

    r = requests.get(f"https://api.spotify.com/v1/audio-features/?ids={r_ids}", headers=headers)
    features = r.json()

    df_r = pd.DataFrame(features["audio_features"])
    df_r["genre"] = "Rock"

    df_combined = pd.concat([df_hh, df_c, df_r])

    df_combined.to_csv(r'song_data.csv', index = False)

    shuffle_data()