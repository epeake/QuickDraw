#!/bin/bash

if [ $(find "/data/scratch/epeake/Google-Doodles/" -name "train.csv") ]
  then
    rm /data/scratch/epeake/Google-Doodles/train.csv
    rm /data/scratch/epeake/Google-Doodles/cross_validate.csv
fi

/home/epeake/venv/mainenv/bin/python3 merge_doodles.py /data/scratch/epeake/Google-Doodles/

# split train and test 80% / 20%
n_lines=$(wc -l /data/scratch/epeake/Google-Doodles/all_doodles.csv | grep -o "[0-9]\+")
n_train=$(expr $n_lines \* 8 / 10)

split -l $n_train /data/scratch/epeake/Google-Doodles/all_doodles.csv /data/scratch/epeake/Google-Doodles/train
mv /data/scratch/epeake/Google-Doodles/trainaa /data/scratch/epeake/Google-Doodles/train.csv
mv /data/scratch/epeake/Google-Doodles/trainab /data/scratch/epeake/Google-Doodles/cross_validate.csv

rm /data/scratch/epeake/Google-Doodles/all_doodles.csv
