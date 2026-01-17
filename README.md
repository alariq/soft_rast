Simple software rasterizer from scratch, no dependencies (see below).
Whole stuff including math & string classes + .obj parser + image loader + simple scene editor and of course rasterizer with perspective correct texture mapping itself :) is ~3k loc (was written ad-hoc for this small project), about similar amount of code to get a triangle on screen in Vulkan :)
This is WIP, no SIMD on purpose as hopefully in future I'll port it to a very tiny MCU

All "interesting" stuff in `main.cpp`

Dependencies :)
------------
None, seriously :) Actual rasterizer does not depend on anything, but I use ImGui for my "editor/debugging framework" (added NOT as submodules for simplicity), so here you are. Probably need to add some `NO_EDITOR` define to strip all this out and save output to .ppm files

Build
-----

Linux:

`cd soft_rast`

`cmake -S . -B ./build`

`cd ./build`

`make -j$(nproc)`

Windows:

never tried, should be simple enough

Test models
-----------
Some test models are converted from nice pack at [elbolilloduro.itch.io](https://elbolilloduro.itch.io/halloween) and [psionicgames.itch.io](https://psionicgames.itch.io/low-poly-space-ship-pack)

Screenshots
-----------
[![Fighter1 model](https://github.com/alariq/soft_rast/raw/master/screenshots/fighter1-shaded.png)]
[![Halloween scene](https://github.com/alariq/soft_rast/raw/master/screenshots/halloween-shaded.png)]

More screenshots are located in the `screenshots` folder
