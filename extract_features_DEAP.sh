#!/usr/bin/env bash

feats_folder="./features_new"
if [ -d $feats_folder ]
then
    echo "Directory ${feats_folder} exists" 
else
    echo "Creating directory ${feats_folder}"
    mkdir -p $feats_folder
fi

for subject_id in `seq -w 1 32`;
do
	python extract_features_DEAP.py --subject_id $subject_id
done

