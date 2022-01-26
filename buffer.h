

struct Params
{


    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
  
    uchar4* frame_buffer;
    int samples;
  
    float3 playerpos;
    float3* verts;

    //uv data

    float4* uvs;
   
    cudaTextureObject_t          tex;
};


struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{

    // No data needed
};
