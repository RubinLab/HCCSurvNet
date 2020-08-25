# HCCSurvNet: Predicting post-surgical recurrence of hepatocellular carcinoma from digital histopathologic images using deep learning  
  
![method_outline](method_outline.png)  

This repository contains the code to quantify risk scores for recurrence in patients with hepatocellular carcinoma from H&E-stained FFPE histopathology images.

## Software Requirements  
This code was developed and tested in the following environment.  
### OS  
- Ubuntu 18.04  
### GPU  
- Nvidia GeForce RTX 2080 Ti  
### Python Dependencies  
- python (3.6.10)  
- numpy: (1.18.1)  
- pandas (0.25.3)  
- pillow (7.0.0)  
- scikit-learn (0.21.3)  
- scikit-image (0.15.0)  
- scikit-survival (0.11)  
- opencv-python (4.1.2.30)  
- openslice-python (1.1.1)  
- staintools (2.1.2)  
- h5py (2.9.0)  
- pytable (3.5.1)  
- pytorch (1.4.0)  
- torchvision (0.5.0)  
  
## Demo  
### Data preparation  
Download diagnostic whole-slide images from [TCGA-LIHC project](https://portal.gdc.cancer.gov/projects/TCGA-LIHC) using [GDC Data Transfer Tool Client](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)  
```
gdc-client download -m gdc_manifest_tcga_lihc.txt
```
  
Download TCGA-CDR-SupplementalTableS1.xlsx from [Integrated TCGA Pan-Cancer Clinical Data Resource](https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018) and rename it to metadata.csv  
  
Get annotations on whole-slide images using [Aperio ImageScope](https://www.leicabiosystems.com/digital-pathology/manage/aperio-imagescope/) in XML format  
  
### Prepaere datasets for tumor tile classification  
```
python xml2tile.py  
python xml_tile2hdf.py  
```
  
### Train tumor tile classifier  
```
python tumor_tile_classifier.py
```
  
### Prepare datasets for tumor tile inference  
```
python svs2tile.py  
python svs_tile2hdf.py  
```
  
### Tumor tile inference  
```
python tumor_tile_inference.py  
```
  
### Select top-X tiles (Default: X==100)  
```
python select_topX.py  
```
  
### Train risk score predictor  
```
python risk_score_predictor.py  
```
  
Note: please edit path in each file  
  
## License  
This code is made available under the MIT License.  
  
## Citation  
Predicting post-surgical recurrence of hepatocellular carcinoma from digital histopathologic images using deep learning  
  
Rikiya Yamashita, Jin Long, Atif Saleem, Daniel Rubin, Jeanne Shen.  