#!/usr/bin/env bash

f_path=$"/data/scratch/epeake/Google-Doodles/"
python_env=$"/home/epeake/venv/mainenv/bin/python3"

$python_env MergeDoodles.py $f_path

sort -R -o $f_path"all_doodles.csv" $f_path"all_doodles.csv"

# split train, and test 80% / 20%
n_lines=$(wc -l $f_path"all_doodles.csv" | grep -o "[0-9]\+")
n_train=$(expr $n_lines \* 8 / 10)

split -l $n_train $f_path"all_doodles.csv" $f_path"train"
mv $f_path"trainaa" $f_path"train_all.csv"
mv $f_path"trainab" $f_path"test.csv"

rm $f_path"all_doodles.csv"
