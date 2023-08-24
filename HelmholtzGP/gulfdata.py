
#small script to look at and engineer gulf data
from tqdm import tqdm
from netCDF4 import Dataset
import numpy as np
ds = Dataset('HelmholtzGP/data/GulfDriftersOpen.nc')
print(ds.data_model)
print()
print(ds)
print()
print(ds.variables.keys())
vars = ['exp_names', 
            'drifter_types',
            'id', 
            'drifter_id',
            'source', 
            'type', 
            'row_size',
            'ids',
            'time', #
            'lat', #
            'lon', #
            'u', #
            'v', #
            'temp', 
            'drogue', 
            'gap', 
            'depth',
            'filled']

for key in vars:
    print(ds.variables[key])
    print()

print()
everyother = 1
onward = 1000000
time = ds.variables['time'][onward:][::everyother]
lat = ds.variables['lat'][onward:][::everyother]
lon = ds.variables['lon'][onward:][::everyother]
u = ds.variables['u'][onward:][::everyother]
v = ds.variables['v'][onward:][::everyother]

#lon - x value, lat - y value
minlon = (-90.1)
maxlon = (-83.9)
minlat = (23.8)
maxlat = (27.5)

nlonbins = 25
nlatbins = int(nlonbins*(maxlat-minlat)/(maxlon-minlon))


loncentres = np.linspace(minlon, maxlon, nlonbins)

latcentres = np.linspace(minlat, maxlat, nlatbins)

binlocs = np.meshgrid(loncentres,latcentres)

binwidth = [loncentres[1]-loncentres[0], latcentres[1]-latcentres[0] ]

gridshape = binlocs[0].shape

ugrid = np.empty(shape = gridshape,dtype = object)
vgrid = np.empty(shape = gridshape, dtype= object)
ubar = np.zeros(shape = gridshape)
vbar = np.zeros(shape = gridshape)

for i in range(nlatbins):
    for j in range(nlonbins):
        ugrid[i,j] = [0]
        vgrid[i,j] = [0]


for i in tqdm(range(int(len(lat)/1))):
    lonindex = np.where((lon[i] < loncentres+binwidth[0]*0.5) & (lon[i] > loncentres-binwidth[0]*0.5) )[0]
    latindex = np.where((lat[i] < latcentres+binwidth[1]*0.5) & (lat[i] > latcentres-binwidth[1]*0.5))[0]
    if not latindex:
        continue
    if not lonindex:
        continue

    ugrid[latindex[0]][lonindex[0]].append(u[i])
    vgrid[latindex[0]][lonindex[0]].append(v[i])

for i in range(nlatbins):
    for j in range(nlonbins):
       ubar[i,j] = np.average(ugrid[i,j])
       vbar[i,j] = np.average(vgrid[i,j])





#does not append to 0th row and 0th column
norowlat = np.delete(binlocs[1], 0, axis = 0)
norowlon = np.delete(binlocs[0], 0, axis = 0)
norowubar = np.delete(ubar,0, axis = 0)
norowvbar = np.delete(vbar,0, axis = 0)

#delete column and flatten
lats = np.delete(norowlat,0, axis = 1).flatten()
lons = np.delete(norowlon,0, axis = 1).flatten()
ubar = np.delete(norowubar,0, axis = 1).flatten()
vbar = np.delete(norowvbar,0, axis = 1).flatten()

shape = [nlonbins-1, nlatbins-1]

#import
import matplotlib.pyplot as plt
#make figure
fig, ax = plt.subplots(1,1)
ax.quiver(lons,lats, ubar ,vbar, color = 'red', label = '', scale = 20)
ax.set(xlim = [minlon-1.0, maxlon+1.0], ylim = [minlat-1.0, maxlat+1.0],)
#plt.savefig('.png', dpi = 600)
plt.show()

ShapeArr = np.zeros(len(lats))
ShapeArr[0] = nlonbins-1
ShapeArr[1] = nlatbins-1

import pandas as pd
df = pd.DataFrame(columns=['lonbins','latbins','ubar','vbar', 'shape'])


df['shape'] = ShapeArr
df['lonbins'] = lons
df['latbins'] = lats
df['ubar'] = ubar
df['vbar'] = vbar




df.to_csv('HelmholtzGP/data/gulfdatacoarse.csv')#, include_index=True)


print()


print()

SomeRows = np.insert(ds.variables['row_size'],0,0)
times = []
lats = []
lons = []
us = []
vs = []
for i in tqdm(range(len(SomeRows)-1)):
    times.append(np.array([time[SomeRows[i]:SomeRows[i+1]]]))
    lats.append(np.array([lat[SomeRows[i]:SomeRows[i+1]]]))
    lons.append(np.array([lon[SomeRows[i]:SomeRows[i+1]]]))
    us.append(np.array([u[SomeRows[i]:SomeRows[i+1]]]))
    vs.append(np.array([v[SomeRows[i]:SomeRows[i+1]]]))

colors = [
    'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray',
    'olive', 'cyan', 'magenta', 'lime', 'teal', 'lavender', 'coral', 'gold',
    'darkblue', 'darkorange', 'darkgreen', 'crimson', 'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray',
    'olive', 'cyan', 'magenta', 'lime', 'teal', 'lavender', 'coral', 'gold',
    'darkblue', 'darkorange', 'darkgreen', 'crimson', 'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray',
    'olive', 'cyan', 'magenta', 'lime', 'teal', 'lavender', 'coral', 'gold',
    'darkblue', 'darkorange', 'darkgreen', 'crimson'
]


#import
import matplotlib.pyplot as plt
#make figure

lons = lons
fig, ax = plt.subplots(1,1)
for i in tqdm(range(len(SomeRows)-1000)):
    ax.quiver(lons[i+400], lats[i+400], us[i+400], vs[i+400], color = colors[i%len(colors)], scale = 20)
plt.show()

ax.quiver()

print()
# Now u_binned and v_binned contain NumPy arrays of u and v values for each bin, respectively.
# lon_bin_locations and lat_bin_locations contain the longitude and latitude bin locations respectively.
arrays = [115, 235, 757]