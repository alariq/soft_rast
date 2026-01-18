#!/bin/sh

ffmpeg -framerate 60 -i ./rec/frame%04d.ppm output.gif
