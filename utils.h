

#include <cuda_runtime.h>



#include <sutil/Camera.h>


#include <iostream>

#ifndef utils
#define utils
//utitlity functions:
inline  float distance(float3 a, float3 b)
{
    float d = sqrt(pow(b.x - a.x, 2) +
        pow(b.y - a.y, 2) +
        pow(b.z - a.z, 2) * 1.0);


    return d;
}
inline  float squaredistance(float3 a, float3 b)
{
    float x = abs(a.x - b.x);
    float y = abs(a.y - b.y);
    float z = abs(a.z - b.z);

    return std::max(z, std::max(x, y));
}
inline  bool Within3DManhattanDistance(float3 c1, float3 c2, float distance)
{
    float dx = abs(c2.x - c1.x);
    float dy = abs(c2.y - c1.y);
    float dz = abs(c2.z - c1.z);

    if (dx > distance) return false; // too far in x direction
    if (dy > distance) return false; // too far in y direction
    if (dz > distance) return false; // too far in z direction

    return true; // we're within the cube
}
inline  float3 getpos(float3 e, int size)
{
    e.x = roundf(e.x / size) * size;
    e.y = roundf(e.y / size) * size;
    e.z = roundf(e.z / size) * size;
    return e;

}


struct compare
{
    float3 key;
    compare(float3 const& i) : key(i) {}

    bool operator()(float3 const& i) {
        return (i.x == key.x && i.y == key.y && i.z == key.z);
    }
};
inline float radians(float degrees)
{
    return degrees * M_PIf / 180.0f;
}
inline float degrees(float radians)
{
    return radians * M_1_PIf * 180.0f;
}

inline bool cnot(float3 a, float3 b) {

    if (a.x == b.x && a.y == b.y && a.z == b.z) {

        return false;
    }
    else {

        return true;
    }

}

inline int getindex(int x, int y, int z, int size) {
    return x + (size + 3) * (y + (size + 3) * z);

}
#endif