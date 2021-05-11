# R2plus1D_S2VT
video captioning about environment information by using R2plus1D and S2VT

# 1. Setting

environment

python <= 3.6  
tensorflow 1.14

```
$ conda create -n tf114 python=3.6
$ conda activate tf114

$ pip install tensorflow-gpu==1.14
$ conda install -c anaconda pandas 
$ conda install -c conda-forge matplotlib
$ conda install -c conda-forge opencv 
$ conda install -c anaconda nltk 
```

workspace

```
$ git clone https://github.com/zzeuui/R2plus1D_S2VT.git
$ cd R2plus1D_S2VT
```

dataset
```
$ wget --no-check-certificate --content-disposition http://hcir.iptime.org/index.php/s/6yezWzgD7ulzmEc/download
$ unzip data.zip
$ rm data.zip
```

# 2. Train and Test

train and test

```
$ python model.py
```
