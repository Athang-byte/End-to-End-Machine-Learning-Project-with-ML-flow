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
AWS CI/CD Deployment with GitHub Actions
1. Login to AWS Console
  
2. Create IAM User for Deployment
   #with specific access
    1.EC2 → Virtual machine for deployment
    2.ECR (Elastic Container Registry) → Store Docker images
   
    #Description : About the Deployment 
    1.Build Docker image of source code
    2.Push Docker image to ECR
    3.Launch EC2 instance
    4.Pull Docker image from ECR into EC2
    5.Run Docker container in EC2
   
    #Policy:
    1.AmazonEC2ContainerRegistryFullAccess
    2.AmazonEC2FullAccess
   
3.Create ECR Repository
-Save the repository URI:<your-account-id>.dkr.ecr.<region>.amazonaws.com/<repository-name>

  
4. Create EC2 machine (Ubuntu)
5. Open EC2 and Install docker in EC2 Machine:
#optinal
sudo apt-get update -y
sudo apt-get upgrade
#required
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
6. Configure EC2 as self-hosted runner:
setting>actions>runner>new self hosted runner> choose os> then run command one by one
7. Setup github secrets:
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=



