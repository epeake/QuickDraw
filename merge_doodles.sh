#!/bin/bash

if [ $(find "/data/scratch/epeake/Google-Doodles/" -name "all_doodles.csv") ]
  then
    rm /data/scratch/epeake/Google-Doodles/all_doodles.csv
fi

/home/epeake/venv/mainenv/bin/python3 merge_doodles.py /data/scratch/epeake/Google-Doodles/
