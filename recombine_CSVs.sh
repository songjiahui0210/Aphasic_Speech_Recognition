#!/bin/bash
# this is to regenerate the whole dataset csv

BASE_DIR="../data_processed/transcripts"

# final csv where all processed csv will be combined
OUTPUT_CSV="../data_processed/clean_dataset.csv"

# temporary directory for storing individual csv file
TEMP_DIR="../data_processed/temp_csvs"

# create the temporary directory if it doesn't exist
mkdir -p $TEMP_DIR

# remove any old output csv file
rm -f $OUTPUT_CSV

# process each folder in the base directory one by one
for folder in "$BASE_DIR"/*/; do        
    OUTPUT_FILE="$folder/clean_dataset.csv"
    
    if [ -f "$OUTPUT_FILE" ]; then
        # move the csv to the temp directory
        cp "$OUTPUT_FILE" "$TEMP_DIR/$(basename $folder)_clean_dataset.csv"
    else
        echo "no csv found in $folder"
    fi
done

# combine all the individual csv into one
echo "combining all csv files into one..."
# add header from the first file
head -n 1 "$(ls "$TEMP_DIR"/*_clean_dataset.csv | head -n 1)" > "$OUTPUT_CSV"
# combine the rest without headers
for file in "$TEMP_DIR"/*_clean_dataset.csv; do
    tail -n +2 "$file" >> "$OUTPUT_CSV"
done

# clean up temporary csv files
rm -rf $TEMP_DIR

echo "All csv have been combined into: $OUTPUT_CSV"
