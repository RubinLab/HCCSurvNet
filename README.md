# HCCSurvNet: Predicting post-surgical recurrence of hepatocellular carcinoma from digital histopathologic images using deep learning  
  
![method_outline](method_outline.png)  

This repository contains the code to quantify risk scores for recurrence in patients with hepatocellular carcinoma from H&E-stained FFPE histopathology images.

The code is written in python.  

## How to use

0. Download diagnostic whole-slide images from [TCGA-LIHC project](https://portal.gdc.cancer.gov/projects/TCGA-LIHC)  
1. Download TCGA-CDR-SupplementalTableS1.xlsx from [here](https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018) and rename it to metadata.csv  
2. Get annotations on whole-slide images using ImageScope in xml format  
3. Run xml2tile.py  
4. Run xml_tile2hdf.py  
5. Run tumor_tile_classifier.py  
6. Run svs2tile.py  
7. Run svs_tile2hdf.py  
8. Run tumor_tile_inference.py  
9. Run select_topX.py  
10. Run risk_score_predictor.py  

## Citation
Predicting post-surgical recurrence of hepatocellular carcinoma from digital histopathologic images using deep learning  

Rikiya Yamashita, Jin Long, Atif Saleem, Daniel Rubin, Jeanne Shen.  