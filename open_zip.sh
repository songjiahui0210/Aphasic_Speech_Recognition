#!/bin/bash

# define useful directory
BASE_DIR="../data_processed"
TRANSCRIPTS_DIR="$BASE_DIR/transcripts"
ZIP_DIR="/work/van-speech-nlp/aphasia/English/Aphasia"

# create the main processed data folder if it doesn't exist
if [ ! -d "$BASE_DIR" ]; then
    mkdir "$BASE_DIR"
    echo "Created directory: $BASE_DIR"
fi

# create the transcripts folder if it does not exist
if [ ! -d "$TRANSCRIPTS_DIR" ]; then
    mkdir "$TRANSCRIPTS_DIR"
    echo "Created directory: $TRANSCRIPTS_DIR"
fi


# process all zip files

for zip_file in "$ZIP_DIR"/*/*.zip; do
    if [ -f "$zip_file" ]; then  # check if the file exists
        # call the open_zip.py script with the current zip file and output directory
        echo "Processing: $zip_file"
        python3 open_zip.py "$zip_file" "$TRANSCRIPTS_DIR" 
    else
        echo "No zip files found in $ZIP_DIR."
    fi
done

echo "Finished processing all zip files."


