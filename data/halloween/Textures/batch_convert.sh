#!/bin/bash
OUTPUT_DIR="tga" # Define your output directory
mkdir -p "$OUTPUT_DIR" # Create the output directory if it doesn't exist

for file in *.jpg *.png *.gif; do # Adjust file extensions as needed
    if [ -f "$file" ]; then # Check if the file exists
        filename=$(basename -- "$file")
        filename_no_ext="${filename%.*}"
        magick "$file" "$OUTPUT_DIR/${filename_no_ext}.tga"
    fi
done
