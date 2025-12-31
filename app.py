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
import datetime
from matplotlib.colors import LinearSegmentedColormap

# ==============================
# 1. Configuration Streamlit
# ==============================
st.set_page_config(layout="wide")
st.title("Analogue atmosphérique Z500")

# ==============================
# 2. Téléchargement du dataset
# ==============================
DATA_FILE = "z500_concatené.nc"
DATA_URL = "https://zenodo.org/records/18102027/files/z500_concaten%C3%A9.nc?download=1"

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
    ds = xr.open_dataset(DATA_FILE)
    z = ds["z"].transpose("time", "latitude", "longitude")
    z = z.sel(longitude=slice(-40, 20), latitude=slice(80, 30))
    return z

z = load_data()
nt, ny, nx = z.shape
st.caption(f"{nt} dates — grille {ny} × {nx}")

# ==============================
# 4. Construction de la matrice X
# ==============================
@st.cache_data
def build_X(_z):
    nt, ny, nx = _z.shape
    return _z.values.reshape(nt, ny * nx)

X = build_X(z)

# ==============================
# 5. Sélecteur de date
# ==============================
min_date = datetime.date(1940, 1, 1)
max_date = datetime.date.today()
date_input = st.date_input(
    "Choisissez une date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

date_user = np.datetime64(date_input)
z_sel = z.sel(time=date_user, method="nearest")
day = int(np.where(z.time.values == z_sel.time.values)[0][0])
date_ref = np.datetime_as_string(z.time[day].values, unit="D")

# ==============================
# 6. Paramètres analogues
# ==============================
N_exclude = 20  # jours autour de la date cible à exclure
N_best = st.slider("Nombre d'analogues à afficher", min_value=1, max_value=10, value=3)

# ==============================
# 7. Calcul des distances
# ==============================
x_ref = X[day]

mask = np.ones(nt, dtype=bool)
mask[max(0, day-N_exclude):min(nt, day+N_exclude+1)] = False

dist = np.full(nt, np.nan)
norm = np.sqrt(X.shape[1])
for i in range(nt):
    if mask[i]:
        dist[i] = np.linalg.norm(X[i] - x_ref) / norm

# ==============================
# 8. Sélection des N meilleurs
# ==============================
idx_sorted = np.argsort(dist)[:N_best]
dates_analog = [np.datetime_as_string(z.time[idx].values, unit="D") for idx in idx_sorted]

st.subheader("Résultats")
st.write("**Jour cible :**", date_ref)
for i, idx in enumerate(idx_sorted):
    st.write(f"**Analogue {i+1} :** {dates_analog[i]} — Distance RMS : {dist[idx]:.2f}")

# ==============================
# 9. Affichage des cartes avec échelle fixe et isolignes
# ==============================
# Colormap violet → rouge
colors = ["violet", "blue", "green", "yellow", "orange", "red"]
cmap = LinearSegmentedColormap.from_list("violet_to_red", colors)

vmin, vmax = 49000, 61000  # échelle fixe

fig, axes = plt.subplots(
    1, N_best + 1, figsize=(5*(N_best+1), 5),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

# Jour cible
ax = axes[0]
pcm = ax.pcolormesh(z.longitude, z.latitude, z.isel(time=day), shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.set_title(f"Jour cible\n{date_ref}")
# tracé des isolignes
cs = ax.contour(z.longitude, z.latitude, z.isel(time=day), levels=np.arange(vmin, vmax+50, 50), colors="black", linewidths=0.7)
ax.clabel(cs, fmt="%d", fontsize=8)
plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05)

# Analogues
for i, idx in enumerate(idx_sorted):
    ax = axes[i+1]
    pcm = ax.pcolormesh(z.longitude, z.latitude, z.isel(time=idx), shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_title(f"Analogue {i+1}\n{dates_analog[i]}")
    # tracé isolignes
    cs = ax.contour(z.longitude, z.latitude, z.isel(time=idx), levels=np.arange(vmin, vmax+50, 50), colors="black", linewidths=0.7)
    ax.clabel(cs, fmt="%d", fontsize=8)
    plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05)

st.pyplot(fig)
