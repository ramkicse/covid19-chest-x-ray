# covid19-chest-x-ray

## Install all dependencies
```
    conda env create -f environment.yml
```

## Download the dataset
```
git clone https://github.com/ieee8023/covid-chestxray-dataset.git
```

```
kaggle competitions download -c rsna-pneumonia-detection-challenge
```

```
mkdir rsna

unzip rsna-pneumonia-detection-challenge.zip -d rsna/
```

## Run the Code

```
jupyter-lab --ip=0.0.0.0
```