#pragma once
#include <cuda_runtime.h>
#include<map>
#include <sutil/Camera.h>
#include "ogt_vox.h"


struct Float3Compare
{
    bool operator() (const float3& lhs, const float3& rhs) const
    {
        return lhs.x < rhs.x
            || (lhs.x == rhs.x && (lhs.y < rhs.y
                || (lhs.y == rhs.y && lhs.z < rhs.z)));
    }
};
class changemanager {
public:

    void registerchange(float4 w);
    std::map<float3, int, Float3Compare> changes;
    float getchange(float3 w);
    void load_model(const ogt_vox_model* model);
};

