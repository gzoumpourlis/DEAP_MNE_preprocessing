#!/usr/bin/env bash

log_folder="./results/log"
if [ -d $log_folder ]
then
    echo "Directory ${log_folder} exists" 
else
    echo "Creating directory ${log_folder}"
    mkdir -p $log_folder
fi

for subject_id in `seq -w 1 32`;
do
	log_file="${log_folder}/s${subject_id}.txt"
	python clean_DEAP.py --subject_id $subject_id | tee $log_file
done

