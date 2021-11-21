# OptixMinecraft
Fully path traced Minecraft using the Optix SDK.

Uses FastNoise: https://github.com/charlesangus/FastNoise

Based off of the OptixTraingle sample file from the SDK.


## Setup
Requires: CUDA, OPTIX SDK, Nvidia RTX gpu

Unzip, put folder in Optix samples directory

Add to the sample directories cmakelists.txt 

Generate CMAKE for optix

Put a 32*16 texture atlas in the debug directory called game.ppm
