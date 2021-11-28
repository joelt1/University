#!/bin/bash -l
#

# Preprocess chosen datasets
for asset in SPXUSD JPXJPY GRXEUR
do
    for year in 2017
    do
        echo "Preprocessing ${asset} ${year}"
        python preprocess_dataset.py $asset $year >> ${asset}_status.txt
        echo "" >> ${asset}_status.txt
        echo "Preprocessed ${asset} ${year}"
        echo ""
    done
done

echo "Selected datasets' preprocessing completed successfully!"
