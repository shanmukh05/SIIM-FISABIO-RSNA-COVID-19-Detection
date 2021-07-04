# SIIM-FISABIO-RSNA-COVID-19-Detection

# Details of the Competition
   - **Objective** : Identify and localize COVID-19 abnormalities on chest radiographs.
   - **Host** : Society for Imaging Informatics in Medicine ([SIIM](https://siim.org/))
   - **Partners** : HP & Intel
   - **Website** : [Kaggle](https://www.kaggle.com/c/siim-covid19-detection/overview/description)
   - **Timeline** : May 18,2021 -> August 10,2021
   - **Evaluation Criteria** : mean Average Precision(mAP) at IoU > 0.5

# Dataset

## Data Structure
- train folder (contains 6300 chest scans in **DICOM** format)
    - study
       - series
           - image
               - .dicom files
- test data   (contains 1200 chest scans in **DICOM** format)
    - study
       - series
           - image
               - .dicom files
-  sample_submission.csv
-  train_image_level.csv
     - id - unique image identifier
     - boxes - bounding boxes in easily-readable dictionary format
     - label - the correct prediction label for the provided bounding boxes
-  train_study_level.csv 
     - id - unique study identifier
     - **Negative for Pneumonia** - 1 if the study is negative for pneumonia, 0 otherwise
     - **Typical Appearance** - 1 if the study has this appearance, 0 otherwise
     - **Indeterminate Appearance**  - 1 if the study has this appearance, 0 otherwise
     - **Atypical Appearance**  - 1 if the study has this appearance, 0 otherwise 


## Data Preprocessing
- **DICOM** images are converted to **JPG** images with various sizes (224,256,512,1024) using the following [Data prepartion](https://github.com/shanmukh05/SIIM-FISABIO-RSNA-COVID-19-Detection/blob/main/notebooks/siim-covid-19-data-preparation.ipynb) notebook.
- Some of the metadata of train and test data is also stored in **csv** files.
- Dataset is uploaded in [Kaggle](https://www.kaggle.com/shanmukh05/siim-covid19-dataset-256px-jpg).

## Data Visualization
- To understand the data more clearly, [data visualization](https://github.com/shanmukh05/SIIM-FISABIO-RSNA-COVID-19-Detection/blob/main/notebooks/siim-covid-19-data-visualization.ipynb) is made using `seaborn`, `matplotlib`, `Pandas Profiling` `wordcloud`.
- Some of the Plots : 

<img src= "https://www.kaggleusercontent.com/kf/64313993/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..0ZKdOw2apQm4F2y7kE-46g.662swcubpc2xE3LhMuyBmI1C3pnXCQ_Q8u2euRn6F60JrUl2JKcod22vr6lvY9wo4oPwRVORUWbXsiWTSOt1q5HACPfhuKRQ8kukSc_ONXLAvH4EdfoiKIeWmy3sZ9qrsClq9mFkrzY2KoFmWeQil94TUVo8OlOCe4aNv2tBrdI731nrq_9oxVMVUUteL1fASVCzVPcpCuhnkkrwwN7qYbJKlgH8g3jM9ESGZ_z0nkOLO0qiG9C07q8-3AGbQJ9tJG91hfRdPMLf9tBYd39ltSg53B70wASaI2Qz018h4ehi1zsnO_iwTJrsQUpKNoIz0KiqQmIJdll62_mYeSHp9JwwWOeR_osMMUblWNA1u-KjoXtLs9_6yf9zsmBscQK7SrP6NLQN0BH1s5R2obNChgaT7LJR90dEHXjii1dAkxnCPHNQSL7QbAswZ4vqJseZ-J95CwFOotLhehxPmTj7tsbIfZ_RMJ5wo4m0vQ85rQQhAB8PL0eL2f8HSC32ysx76ZH5VR__9mnQoV9ShonJMVzAEHne8b6KQPyz9EPQ9jy8ljIdJYcFHV1laJb-Pri2O-wATp6tSWfxx6dRm2zrYLH1umLNKJ--UpZ8IUrR_pWPoiTxclJ_5B6-4T_LgRSBxtN8nDD4oXI4uIwrF1eP7Afxeno7qQuOpPDKg7_bMTAPcTmixO157yXlZwFQRNFN.k9dVN_Mdo1hyQONg0jZhPg/__results___files/__results___12_1.png" width="250" height="250">    <img src="https://www.kaggleusercontent.com/kf/64313993/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..0ZKdOw2apQm4F2y7kE-46g.662swcubpc2xE3LhMuyBmI1C3pnXCQ_Q8u2euRn6F60JrUl2JKcod22vr6lvY9wo4oPwRVORUWbXsiWTSOt1q5HACPfhuKRQ8kukSc_ONXLAvH4EdfoiKIeWmy3sZ9qrsClq9mFkrzY2KoFmWeQil94TUVo8OlOCe4aNv2tBrdI731nrq_9oxVMVUUteL1fASVCzVPcpCuhnkkrwwN7qYbJKlgH8g3jM9ESGZ_z0nkOLO0qiG9C07q8-3AGbQJ9tJG91hfRdPMLf9tBYd39ltSg53B70wASaI2Qz018h4ehi1zsnO_iwTJrsQUpKNoIz0KiqQmIJdll62_mYeSHp9JwwWOeR_osMMUblWNA1u-KjoXtLs9_6yf9zsmBscQK7SrP6NLQN0BH1s5R2obNChgaT7LJR90dEHXjii1dAkxnCPHNQSL7QbAswZ4vqJseZ-J95CwFOotLhehxPmTj7tsbIfZ_RMJ5wo4m0vQ85rQQhAB8PL0eL2f8HSC32ysx76ZH5VR__9mnQoV9ShonJMVzAEHne8b6KQPyz9EPQ9jy8ljIdJYcFHV1laJb-Pri2O-wATp6tSWfxx6dRm2zrYLH1umLNKJ--UpZ8IUrR_pWPoiTxclJ_5B6-4T_LgRSBxtN8nDD4oXI4uIwrF1eP7Afxeno7qQuOpPDKg7_bMTAPcTmixO157yXlZwFQRNFN.k9dVN_Mdo1hyQONg0jZhPg/__results___files/__results___19_1.png" width="250" height="250">   <img src = "https://www.kaggleusercontent.com/kf/64313993/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..0ZKdOw2apQm4F2y7kE-46g.662swcubpc2xE3LhMuyBmI1C3pnXCQ_Q8u2euRn6F60JrUl2JKcod22vr6lvY9wo4oPwRVORUWbXsiWTSOt1q5HACPfhuKRQ8kukSc_ONXLAvH4EdfoiKIeWmy3sZ9qrsClq9mFkrzY2KoFmWeQil94TUVo8OlOCe4aNv2tBrdI731nrq_9oxVMVUUteL1fASVCzVPcpCuhnkkrwwN7qYbJKlgH8g3jM9ESGZ_z0nkOLO0qiG9C07q8-3AGbQJ9tJG91hfRdPMLf9tBYd39ltSg53B70wASaI2Qz018h4ehi1zsnO_iwTJrsQUpKNoIz0KiqQmIJdll62_mYeSHp9JwwWOeR_osMMUblWNA1u-KjoXtLs9_6yf9zsmBscQK7SrP6NLQN0BH1s5R2obNChgaT7LJR90dEHXjii1dAkxnCPHNQSL7QbAswZ4vqJseZ-J95CwFOotLhehxPmTj7tsbIfZ_RMJ5wo4m0vQ85rQQhAB8PL0eL2f8HSC32ysx76ZH5VR__9mnQoV9ShonJMVzAEHne8b6KQPyz9EPQ9jy8ljIdJYcFHV1laJb-Pri2O-wATp6tSWfxx6dRm2zrYLH1umLNKJ--UpZ8IUrR_pWPoiTxclJ_5B6-4T_LgRSBxtN8nDD4oXI4uIwrF1eP7Afxeno7qQuOpPDKg7_bMTAPcTmixO157yXlZwFQRNFN.k9dVN_Mdo1hyQONg0jZhPg/__results___files/__results___43_0.png" width="250" height="250"> 

# Training

## Study Level

- Study Level prediction is a multi class classification task i.e., we have to predict whether given chest x-ray belongs to one of the `Negative for Pneumonia`, `Typical Appearance`, `Indeterminate Appearance`, `Atypical Appearance` categories.
- TensorFlow pretrained models, ChexNet model, various image sizes are used for experimentation.
- More details are in [Study Level Prediction](https://github.com/shanmukh05/SIIM-FISABIO-RSNA-COVID-19-Detection/blob/main/notebooks/siim-covid-19-study-level-predictions.ipynb) notebook.
- Best models are saved in `.h5` format.

## Image Level

- Image Level prediction is a object detection task where we have to localize the abnormality in chest x-rays.
- I used YOLOv5 to train the model using various image sizes and cross validation techniques.
- More details of training are in [Image Level Prediction](https://github.com/shanmukh05/SIIM-FISABIO-RSNA-COVID-19-Detection/blob/main/notebooks/siim-covid-19-yolo-v5-image-level-predictions.ipynb) notebook.
- Best models are saved in `.pt` format.
- Training results are examined using [Wieghts & Biases](https://wandb.ai/shanmukh/siim_covid19_yolov5/reports/SIIM-COVID19-Image-Level-Predictions--Vmlldzo3MzI3MjQ) dashboard.
- To know more about Object detection and it's evolution over years, read research papers present in [papers](https://github.com/shanmukh05/SIIM-FISABIO-RSNA-COVID-19-Detection/tree/main/papers) folder.

# Inference

- Finally the saved models from Study Level and Image Level are loaded and used them predict result on unseen test data which contains over 1200 chest x-rays.
- More details of Inference are in [Inference](https://github.com/shanmukh05/SIIM-FISABIO-RSNA-COVID-19-Detection/blob/main/notebooks/siim-covid-19-final-inference.ipynb) notebook.
