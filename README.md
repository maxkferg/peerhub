# Peerhub Image Challenge


## Build the combined dataset
python model.py --prepare "/Volumes/Flash/Data/PHI/t1"

## Training locally
```sh
python model.py --dataset "/Volumes/Flash/Data/PHI/t1"
```


## Training on Google Cloud

Connect to the Google Cloud
```sh
gcloud config set project stanford-projects
gcloud compute ssh --zone "us-west1-b" "maxkferg@mobile-robot-training"
```

Train on the Google Cloud
```sh
# Prepare dataset
python model.py --prepare /home/maxkferg/data/PHI/

# Train on combined dataset
python model.py --dataset /home/maxkferg/data/PHI/combined
```

Copy the datafiles to the google cloud
```sh
gcloud compute scp --recurse --zone "us-west1-b" "/Volumes/Flash/Data/PHI/t1"  "maxkferg@mobile-robot-training:data/PHI/"
```