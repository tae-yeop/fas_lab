#!/bin/bash

SRC="/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/train"
DEST="/purestorage/project/tyk/3_CUProjects/FAS/dreambooth-moire/original/uhdm2"

find "$SRC" -type f -name "*.jpg" | parallel -j 10 bash -c '
    file={}
    dirname=$(basename "$(dirname "$file")")
    filename=$(basename "$file")
    cp "$file" "'"$DEST"'/${dirname}_$filename"
'
