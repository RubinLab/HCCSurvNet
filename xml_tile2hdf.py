import pandas as pd
import glob
import random
import os
import cv2
import numpy as np
import tables

xmltile_path = '/path/to/xml_tiles/'
hdf5_path = 'path/to/save.hdf5'
csv = 'path/to/metadata.csv'

df = pd.read_csv(csv)
df_xml['ids']=df_xml['tile_name'].str[:23]
df=df.rename(columns={'tile_name':'filename'})

tiles = glob.glob(xml2tile_path+'*/*.png')

labels = []
ids = []

for j in tiles:
    i = os.path.split(j)[1]
    label = df[df.filename==i].label.values[0]
    labels.append(label)
    idx = df[df.filename==i].ids.values[0]
    ids.append(idx)

img_dtype = tables.UInt8Atom()
data_shape = (0, 1024, 1024, 3)
hdf5_file = tables.open_file(hdf5_path, mode='w')
storage = hdf5_file.create_earray(hdf5_file.root, 'img', img_dtype, shape=data_shape)
hdf5_file.create_array(hdf5_file.root, 'labels', labels)
hdf5_file.create_array(hdf5_file.root, 'ids', ids)

for i in range(len(tiles)):
    if i % 1000 == 0 and i > 1:
        print('data: {}/{}'.format(i, len(test_addrs)))

    tile = tiles[i]
    try:
        img = cv2.imread(tile)
        if img.shape[0]!=1024 or img.shape[1]!=1024:
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        storage.append(img[None])

hdf5_file.close()