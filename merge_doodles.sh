#!/bin/bash

f_path=$"/data/scratch/epeake/Google-Doodles/"
python_env=$"/home/epeake/venv/mainenv/bin/python3"

if [ $(find $f_path -name "train.csv") ]
  then
    rm $f_path"train.csv"
    rm $f_path"cross_validate.csv"
fi

$python_env merge_doodles.py $f_path

# split train and test 80% / 20%
n_lines=$(wc -l $f_path"all_doodles.csv" | grep -o "[0-9]\+")
n_train=$(expr $n_lines \* 8 / 10)

split -l $n_train $f_path"all_doodles.csv" $f_path"train"
mv $f_path"trainaa" $f_path"train.csv"
mv $f_path"trainab" $f_path"cross_validate.csv"

rm $f_path"all_doodles.csv"

