import pandas as pd
import glob
import random
import os
import cv2
import numpy as np
import tables

img_path = 'path/to/svs_tiles/'
hdf5_path = 'path/to/save/tumor_tile_inference.hdf5'
csv = 'path/to/tcga_metadata.csv' # please download from https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018

df = pd.read_csv(csv)

ids = df.bcr_patient_barcode.values.tolist()
random.seed(218)

print('Start Glob.Glob')
all_tiles = glob.glob(img_path+'*/*.png')
print('End Glob.Glob')

tiles = []
ids = []
slides = []
fnames = []
pfi = []
pfitime = []

for n, j in enumerate(all_tiles):
    if n % 1000 == 0 and n > 1:
        print('Done: {}/{}'.format(n, len(all_tiles)))
    
    idx = '-'.join(os.path.split(j)[1].split('.')[0].split('-')[:-3])
    slide = os.path.split(j)[1].split('.')[0]
    fname = os.path.basename(j)
    pfi = df[df.bcr_patient_barcode==idx].PFI.values[0]
    pfitime = df[df.bcr_patient_barcode==idx]['PFI.time'].values[0]

    tiles.append(j)
    ids.append(idx)
    slides.append(slide)
    fnames.append(fname)
    pfi.append(pfi)
    pfitime.append(pfitime)

img_dtype = tables.UInt8Atom()
data_shape = (0, 1024, 1024, 3)
# open a hdf5 file and create earrays
hdf5_file = tables.open_file(hdf5_path, mode='w')
storage = hdf5_file.create_earray(hdf5_file.root, 'img', img_dtype, shape=data_shape)

err=[]

for i in range(len(all_tiles)):
    if i % 1000 == 0 and i > 1:
        print('data: {}/{}'.format(i, len(all_tiles)))

    tile = all_tiles[i]
    try:
        img = cv2.imread(tile)
        if img.shape[0]!=1024 or img.shape[1]!=1024:
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        storage.append(img[None])
    except:
        err.append(tile)
        
dellist = lambda items, indexes: [item for index, item in enumerate(items) if index not in indexes]

if len(err)!=0:
    idx = [tiles.index(i) for i in err]
    
    tiles = dellist(tiles, idx)    
    ids = dellist(ids, idx)
    slides = dellist(slides, idx)
    fnames = dellist(fnames, idx)    
    pfi = dellist(pfi, idx)
    pfitime = dellist(pfitime, idx)
  
hdf5_file.create_array(hdf5_file.root, 'ids', ids)
hdf5_file.create_array(hdf5_file.root, 'slides', slides)
hdf5_file.create_array(hdf5_file.root, 'fnames', fnames)
hdf5_file.create_array(hdf5_file.root, 'test_pfi', test_pfi)
hdf5_file.create_array(hdf5_file.root, 'test_pfitime', test_pfitime)

hdf5_file.close()