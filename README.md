Simple software rasterizer from scratch, no dependencies (see below).
Whole stuff including math & string classes + .obj parser + simple scene editor and of course rasterizer itself :) is ~3k loc, similar to what it itakes to get a triangle on screen in vulkan :)
This is WIP, no SIMD on purpose as hopefully in future I'll port it to a very tiny MCU

All "interesting" stuff in `main.cpp`

Dependencies :)
------------
None, seriously :) Actual rasterizer does not depend on anything, but I use ImGui for my "editor/debugging framework", so here you are. Probably need to add some `NO_EDITOR` define to strip all this out and save output to .ppm files

Build
-----

Linux:

`cd soft_rast`
`cmake -S . -B ./build`
`cd ./build`
`make -j$(nproc)`

Windows:

never tried, should be simple enough

Data
----
some test models are converted from nice pack taken from [here](https://elbolilloduro.itch.io/halloween)
