# Neuromorphic YOLO - CVPR 24 (SynSense)

ANN and SNN models for Eye Tracking on SynSense

## Getting started

### Clone the repo

```
git clone https://gitlab.ethz.ch/pbonazzi/eye-tracking-synsense.git
cd eye-tracking-synsense
```


### Create the environment

```
python3.10 -m venv venv
source venv/bin/activate

pip install pandas wandb fire
pip install tonic sinabs sinabs-dynapcnn sinabs-exodus
pip install git+https://gitlab.com/inivation/dv/dv-processing #https://dv-processing.inivation.com/rel_1.7/installation.html
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
```


### Get the dataset


Verify `EyeTrackingDataSet_FromInivation`

```
.
├── 2022-04-12_adam_01
│   ├── annotations.csv
│   └── events.aedat4
├── ...
├── silver.csv
```


## Training

```
python train_snn.py 
```


## Authors and acknowledgment
Author : Pietro Bonazzi.