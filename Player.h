#pragma once
#include <cuda_runtime.h>

//player class
class Player {
public:
    float3 moveto = { 0,0,0 };
    float3 playerpos = { 0,0,2.0f };
    float vz = 0;

    void update(float delta);
};


