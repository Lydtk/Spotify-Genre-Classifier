import pandas as pd
import numpy as np
import requests 
import json 
from pprint import pprint

f = open('credentials.json',)
creds = json.load(f)
CLIENT_ID = creds["client_id"]
CLIENT_SECRET = creds["client_secret"]
f.close()

body_params = {"grant_type" : "client_credentials"}

url='https://accounts.spotify.com/api/token'

response = requests.post(url, data=body_params, auth = (CLIENT_ID, CLIENT_SECRET))

token = response.json()["access_token"]
pprint(token)

headers = {'Authorization': 'Bearer '+ token}

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
#hh_ids = hh_ids.join()