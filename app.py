
import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

st.set_page_config(layout="wide")
st.title("Analogue atmosphérique Z500")

# Chargement lazy du dataset
@st.cache_data
def load_data():
    ds = xr.open_dataset("z500_concatené.nc")
    z = ds["z"].transpose("time", "latitude", "longitude")
    z = z.sel(longitude=slice(-40, 20), latitude=slice(80, 30))
    return z

z = load_data()

nt, ny, nx = z.shape
X = z.values.reshape(nt, ny * nx)

# Sélecteur de date
date_input = st.date_input("Choisissez une date")

# Conversion
date_user = np.datetime64(date_input)

z_sel = z.sel(time=date_user, method="nearest")
day = int(np.where(z.time.values == z_sel.time.values)[0][0])

# Calcul meilleur analogue
N = st.slider("Jours exclus autour de la date", 1, 10, 5)

x_ref = X[day]

mask = np.ones(nt, dtype=bool)
mask[max(0, day-N):min(nt, day+N+1)] = False

dist = np.full(nt, np.nan)
for i in range(nt):
    if mask[i]:
        dist[i] = np.linalg.norm(X[i] - x_ref) / np.sqrt(X.shape[1])

idx_best = np.nanargmin(dist)

date_ref = np.datetime_as_string(z.time[day].values, unit="D")
date_analog = np.datetime_as_string(z.time[idx_best].values, unit="D")

st.write("### Résultat")
st.write("Jour cible :", date_ref)
st.write("Meilleur analogue :", date_analog)
st.write("Distance RMS :", dist[idx_best])

# Affichage cartes
fig, axes = plt.subplots(1, 2, figsize=(14,6),
                         subplot_kw={"projection": ccrs.PlateCarree()})

ax = axes[0]
pcm = ax.pcolormesh(z.longitude, z.latitude, z.isel(time=day))
ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.set_title(f"Jour cible\n{date_ref}")
plt.colorbar(pcm, ax=ax, orientation="horizontal")

ax = axes[1]
pcm = ax.pcolormesh(z.longitude, z.latitude, z.isel(time=idx_best))
ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.set_title(f"Meilleur analogue\n{date_analog}")
plt.colorbar(pcm, ax=ax, orientation="horizontal")

st.pyplot(fig)
