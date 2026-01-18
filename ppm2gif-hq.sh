#!/bin/sh


#usage: ppm2gif-hq.sh <out.gif>

palette="/tmp/palette.png"
mkv_tmp="/tmp/o.mkv"
filters="fps=60,scale=320:-1:flags=lanczos"


# create mkv from image list
ffmpeg -framerate 60 -i ./rec/frame%04d.ppm ${mkv_tmp}
ffmpeg -v warning -i ${mkv_tmp} -vf "$filters,palettegen" -y $palette
ffmpeg -v warning -i ${mkv_tmp} -i $palette -lavfi "$filters [x]; [x][1:v] paletteuse" -y $1

#ffmpeg -v warning -i $1 -vf "$filters,palettegen" -y $palette
#ffmpeg -v warning -i $1 -i $palette -lavfi "$filters [x]; [x][1:v] paletteuse" -y $2

