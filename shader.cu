
#include <optix.h>


#include "buffer.h"
#include "random.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>

extern "C" {
    __constant__ Params params;
}

# define M_PI           3.14159265358979323846

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

struct RadiancePRD
{
    // TODO: move some state directly into payload registers?
    float3       emitted;
    float3       radiance;
    float3       attenuation;
    float3       origin;
    float3       direction;
    unsigned int seed;
 
    int          done;
    int          pad;
};


struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}


static __forceinline__ __device__ void  packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ RadiancePRD* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ bool get_face_normal(float3 dir, float3 outward_normal) {
    bool front_face = dot(dir, outward_normal) < 0;
    return front_face;
}
static __forceinline__ __device__ void setPayloadOcclusion(bool occluded)
{
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}


static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}




static __forceinline__ __device__ float3 refract(float3 uv, float3 n, float etai_over_etat) {
    float cos_theta = min(dot(uv * make_float3(-1), n), 1.0f);
    float3 r_out_perp = make_float3(etai_over_etat) * (uv + make_float3(cos_theta) * n);
    float3 r_out_parallel = make_float3(-sqrt(fabs(1.0f - pow(length(r_out_perp), 2.0f)))) * n;
    return r_out_perp + r_out_parallel;
}


static __forceinline__ __device__ float distancef(float3 a, float3 b)
{
    float3 v = b - a;
    return sqrt(dot(v, v));






    
}

static __forceinline__ __device__ float reflectance(float cosine, float ref_idx) {
    // Use Schlick's approximation for reflectance.
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5.0f);
}
static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    RadiancePRD* prd
)
{
    // TODO: deduce stride from num ray-types passed in params

    unsigned int u0, u1;
    packPointer(prd, u0, u1);
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0,        // SBT offset
        1,           // SBT stride
        0,        // missSBTIndex
        u0, u1);
}




//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const int    w = params.image_width;
    const int    h = params.image_height;
    const float3 eye = params.cam_eye;
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const uint3  idx = optixGetLaunchIndex();
  

    unsigned int seed = tea<4>(idx.y * w + idx.x, 0);

    float3 result = make_float3(0.0f);

    int samples = params.samples;
    int i = samples;
    do
    {
        // The center of each pixel is at fraction (0.5,0.5)
        const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

        const float2 d = 2.0f * make_float2(
            (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(w),
            (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(h)
        ) - 1.0f;

        float3 ray_direction = normalize(d.x * U + d.y * V + W);
        float3 ray_origin = eye;

        RadiancePRD prd;
        prd.emitted = make_float3(0.f);
        prd.radiance = make_float3(0.f);
        prd.attenuation = make_float3(1.f);
    
        prd.done = false;
        prd.seed = seed;

        int depth = 0;
        for (;; )
        {
            traceRadiance(
                params.handle,
                ray_origin,
                ray_direction,
                0.01f,  // tmin       // TODO: smarter offset
                1e16f,  // tmax
                &prd);

         
            result += prd.radiance * prd.attenuation;

            if (prd.done || depth >= 3) // TODO RR, variable for depth
                break;

            ray_origin = prd.origin;
            ray_direction = prd.direction;

            ++depth;
        }
    } while (--i);

    const uint3    launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.image_width + launch_index.x;
    float3         accum_color = result / static_cast<float>(samples);

 
    params.image[idx.y * params.image_width + idx.x] = make_color(accum_color);
}


extern "C" __global__ void __miss__ms()
{
    MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
  ;
    RadiancePRD* prd = getPRD();

    prd->radiance =miss_data->bg_color;

  //  float3 ray_dir = optixGetWorldRayDirection();

 //   if (ray_dir.x > 0.5 && ray_dir.x < 0.6 && ray_dir.y > 0.5 && ray_dir.y < 0.6) {
   //     prd->attenuation = make_float3(20000, 20000, 20000);
  //      prd->radiance = make_float3(1, 1, 1);
   // }
  

   //    float3 ray_dir = optixGetWorldRayDirection();

 //   if (ray_dir.z < -0.05 ) {
      
  //      prd->radiance = make_float3(0, 0, 0);
 //   }

    //debug override
 //   prd->radiance = {0.01,0.01,0.01};
    prd->done = true;
}



// C:\Users\Philip\Documents\Sketchfab_2020_07_11_17_10_282.blend.rts

extern "C" __global__ void __closesthit__ch()
{

    const int    prim_idx = optixGetPrimitiveIndex();
    const float3 ray_dir = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx * 3;


    

    float3 origin = optixGetWorldRayOrigin();





    RadiancePRD* prd = getPRD();
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    if (distancef(params.playerpos, P) > 100) {


        prd->radiance = make_float3(0.3f);
        prd->attenuation = { 1,1,1 };
        prd->done = true;

        return ;
    }
    


   




    float3 v0 = params.verts[vert_idx_offset];

    float3 v1 = params.verts[vert_idx_offset+1];

    float3 v2 = params.verts[vert_idx_offset+2];



 



        //mat color
        const float2 barycentrics = optixGetTriangleBarycentrics();
        float3 uv = make_float3(barycentrics, 1.0f - barycentrics.x - barycentrics.y);
     
        float2 uv1 = make_float2(params.uvs[vert_idx_offset]);
        float2 uv2  = make_float2(params.uvs[vert_idx_offset+1]);
        float2 uv3 = make_float2(params.uvs[vert_idx_offset+2]);
        // float2 uv1 = { 0,1 };
    //   float2 uv2 = { 0,0 };
    //    float2 uv3 = { 1,1 };
        float2 uvv = uv.x * uv1 + uv.y * uv2 + uv.z * uv3;

        float4 fromTexture = tex2D<float4>(params.tex, uvv.x, uvv.y);


        float3 col = make_float3(fromTexture);




  


        const float3 N_0 = normalize(cross(v1 - v0, v2 - v0));

        const float3 N = faceforward(N_0, -ray_dir, N_0);




        int mat = params.uvs[vert_idx_offset].z;

        unsigned int seed = prd->seed;


        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in;
        cosine_sample_hemisphere(z1, z2, w_in);
        Onb onb(N);
        onb.inverse_transform(w_in);


        prd->direction = w_in;
     //   prd->direction = normalize(w_in);
        

        if (mat == -2) {
            if (col.x > 0.9) {
               // col = { 0,0.5,0 };
           //     prd->radiance = { 1,1,1 };
                prd->direction = ray_dir;
            }


        }
      


        if (mat == 1) {

            float3 reflected = reflect(normalize(ray_dir), N) + make_float3(0.1) * w_in;








            prd->direction = reflected;
        }
       


        if (mat == 6) {

         //   if (col.x < 0.5) {

         //   }


                col = { 0.9,0.9,0.9 };
                bool inorout = get_face_normal(ray_dir, N);
                float ir = 1.5f;
                float refraction_ratio = inorout ? (1.0 / ir) : ir;
                float cos_theta = min(dot(normalize(ray_dir) * make_float3(-1), N), 1.0);
                float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
                bool cannot_refract = (refraction_ratio * sin_theta) > 1.0;
                float3 reflected;
                if (cannot_refract || reflectance(cos_theta, refraction_ratio) > z1) {
                    reflected = reflect(normalize(ray_dir), N);


                }

                else {

                    reflected = refract(normalize(ray_dir), N, refraction_ratio);
                }

                prd->direction = reflected;
            
        }






        prd->origin = P;








        /*
        if (v1.z > 4) {

            col *= 10;
        }*/

        /*  if (distancef(params.playerpos, P) < 2) {

              col *= 1000;
          }*/
          //checker tex


          //size of tex(small makes kind of rough)
        if (mat == 2) {
            //float dist = distancef(P, origin);
        //    prd->radiance = { 1 + 10/dist , 1 + 10 / dist, 1 + 10 / dist };
            prd->radiance = { 2 , 2,2};
            prd->done = true;
            return;
        }




        prd->attenuation *= col ;
   
    
        if (distancef(params.playerpos, P) > 80) {

            float distdiv = max((distancef(params.playerpos, P)-80)*2,1.0f);
            col = col * distdiv;
            prd->radiance = make_float3(min(col.x , 1.0f), min(col.y, 1.0f), min(col.z, 1.0f)) * make_float3(0.3f);
            prd->attenuation = { 1,1,1 };
            prd->done = true;
        }
      
   
   
}
