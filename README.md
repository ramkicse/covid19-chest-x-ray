# Diagnosing COVID-19 using AI-based medical Image Analyses

![COVID-19!](https://www2.deloitte.com/content/dam/insights/us/articles/6677_covid-19/images/6677_banner1.jpg/_jcr_content/renditions/cq5dam.web.1440.660.jpeg)
![version!](https://img.shields.io/badge/covid19--chest--x--ray-1.0.0-blue)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ramkicse/covid19-chest-x-ray/master)

## Install all dependencies
    conda env create -f environment.yml


## Download the dataset
    git clone https://github.com/ieee8023/covid-chestxray-dataset.git


    kaggle competitions download -c rsna-pneumonia-detection-challenge
    mkdir rsna

    unzip rsna-pneumonia-detection-challenge.zip -d rsna/


## Run the Code

    jupyter-lab --ip=0.0.0.0
