#pragma once
#include <cuda_runtime.h>
#include <sutil/sutil.h>
#include <sutil/GLDisplay.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <mutex>
#include "utils.h"
//classes
//manager class
class Vertexmanager {     
public:           
    std::vector<float3> vertices;
    std::vector<float3> newvertices;

    std::vector<float4> uvs;
    std::vector<float4> newuvs;
    std::mutex lock;
    std::vector<float3> index;
    std::vector<float3> newindex;
    void removechunk(float3 pos);
    void cull(float3 pos, std::vector <float3> &chunks);
  
    void updatenew();
    void add(float3 pos, float4 uvs, float3 i);
    void addtri(float3 a, float3 b, float3 c, float4 uva, float2 uvb, float2 uvc, float3 i);
    void addplane(float3 a, float3 b, float3 c, float3 d, float mat, float3 i, int side);
    int viewdist;
    int halfsize;
    int size;
   
   


};

