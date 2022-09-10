#!/bin/bash

echo 'downloading ICEWS14 data source'
wget https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/28075/K7L9Y8 -O ./data/raw/ICEWS14/icews14_data_source.tab

mkdir ./data/processed
# WN18RR and FB15k-237 are downloaded from https://github.com/wangbo9719/StAR_KGC
# ICEWS14 is downloaded from https://github.com/mniepert/mmkb
# NELL-One is the reformatted data from https://github.com/wangbo9719/StAR_KGC
python ./script/process_wn18rr.py
python ./script/process_fb15k237.py
python ./script/process_fb15k237n.py
python ./script/process_icews14.py
python ./script/process_nell.py

echo 'done!'