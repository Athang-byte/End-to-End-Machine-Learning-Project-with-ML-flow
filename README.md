# End-to-End-Machine-Learning-Project-with-ML-flow


## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the app.py




# How to run?
### STEPS:
Clone the repository
bash
https://github.com/Athang-byte/End-to-End-Machine-Learning-Project-with-ML-flow flow

### STEP 01- Create a conda environment after opening the repository

```bash

conda create -n mlproj python=3.8 -y
```

```bash
conda activate mlproj
```

### STEP 02- install the requirements 
```bash
pip install -r requirements.txt 
```


```bash
You, 3 seconds ago Uncommitted changes
#Finally run the following command
python app.py
```
Now,
```bash
open up you local host and port
```
You, 2 seconds ago | 1 author (You)

## MLflow
[Documentation](https://mlflow.org/docs/latest/index.html)
You, 5 seconds ago | 1 author (You)

##### cmd
- mlflow ui
### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/Athang-byte/End-to-End-Machine-Learning-Project-with-ML-flow.mlflow \
MLFLOW_TRACKING_USERNAME=Athang-byte \
MLFLOW_TRACKING_PASSWORD=25168ee471fca5d7644ce6d4f2fe954248f25e69 \
python script.py 

Run this to export as env variables:

```bash

$env:MLFLOW_TRACKING_URI="https://dagshub.com/Athang-byte/End-to-End-Machine-Learning-Project-with-ML-flow.mlflow"
$env:MLFLOW_TRACKING_USERNAME="Athang-byte"
$env:MLFLOW_TRACKING_PASSWORD="25168ee471fca5d7644ce6d4f2fe954248f25e69"
```