# ==============================
# 0. Imports
# ==============================
import os
import requests
import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==============================
# 1. Configuration Streamlit
# ==============================
st.set_page_config(layout="wide")
st.title("Analogue atmosphérique Z500")

# ==============================
# 2. Téléchargement du dataset (1 seule fois)
# ==============================
DATA_FILE = "z500_concatené.nc"
DATA_URL = "https://zenodo.org/record/1234567/files/z500_concatené.nc"

@st.cache_resource
def download_data():
    if not os.path.exists(DATA_FILE):
        with st.spinner("Téléchargement du dataset Z500..."):
            r = requests.get(DATA_URL, stream=True)
            r.raise_for_status()
            with open(DATA_FILE, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return DATA_FILE

download_data()

# ==============================
# 3. Chargement lazy du dataset
# ==============================
@st.cache_data
def load_data():
    ds = xr.open_dataset(DATA_FILE)  # lazy
    z = ds["z"].transpose("time", "latitude", "longitude")
    z = z.sel(
        longitude=slice(-40, 20),
        latitude=slice(80, 30)
    )
    return z

z = load_data()

nt = z.sizes["time"]
ny = z.sizes["latitude"]
nx = z.sizes["longitude"]

st.caption(f"{nt} dates — grille {ny} × {nx}")

# ==============================
# 4. Construction de la matrice X (UNE seule fois)
# ==============================
@st.cache_data
def build_X(z):
    nt, ny, nx = z.shape
    return z.values.reshape(nt, ny * nx)

X = build_X(z)

# ==============================
# 5. Sélection de la date utilisateur
# ==============================
date_input = st.date_input("Choisissez une date")
date_user = np.datetime64(date_input)

z_sel = z.sel(time=date_user, method="nearest")
day = int(np.where(z.time.values == z_sel.time.values)[0][0])

date_ref = np.datetime_as_string(z.time[day].values, unit="D")

# ==============================
# 6. Paramètres analogue
# ==============================
N = st.slider(
    "Nombre de jours exclus autour de la date cible",
    min_value=1,
    max_value=10,
    value=5
)

# ==============================
# 7. Calcul du meilleur analogue
# ==============================
x_ref = X[day]

mask = np.ones(nt, dtype=bool)
mask[max(0, day - N):min(nt, day + N + 1)] = False

dist = np.full(nt, np.nan)
norm = np.sqrt(X.shape[1])

for i in range(nt):
    if mask[i]:
        dist[i] = np.linalg.norm(X[i] - x_ref) / norm

idx_best = np.nanargmin(dist)
date_analog = np.datetime_as_string(z.time[idx_best].values, unit="D")

# ==============================
# 8. Résultats texte
# ==============================
st.subheader("Résultat")
st.write("**Jour cible :**", date_ref)
st.write("**Meilleur analogue :**", date_analog)
st.write("**Distance RMS :**", f"{dist[idx_best]:.2f}")

# ==============================
# 9. Affichage des cartes
# ==============================
fig, axes = plt.subplots(
    1, 2, figsize=(14, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

# Jour cible
ax = axes[0]
pcm = ax.pcolormesh(
    z.longitude,
    z.latitude,
    z.isel(time=day),
    shading="nearest"
)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.set_title(f"Jour cible\n{date_ref}")
plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05)

# Analogue
ax = axes[1]
pcm = ax.pcolormesh(
    z.longitude,
    z.latitude,
    z.isel(time=idx_best),
    shading="nearest"
)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.set_title(f"Meilleur analogue\n{date_analog}")
plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05)

st.pyplot(fig)
