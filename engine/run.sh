#!/bin/bash

cats=("bathhub" "bed" "bookcase" "cabinet" "sofa")
modes=("test" "train")

for cat in "${cats[@]}";do
	python eval.py --per_obj $cat 
done

