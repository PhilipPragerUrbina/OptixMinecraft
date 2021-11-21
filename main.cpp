
#include <glad/glad.h> 

#include <cuda_gl_interop.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sutil/GLDisplay.h>
#include <cuda_runtime.h>
#include <numeric>
#include <sampleConfig.h>
#include <sutil/sutil.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <map>
#include <GLFW/glfw3.h>
#include "buffer.h"
#include <sutil/Exception.h>
#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <sutil/Camera.h>
#include <sutil/Trackball.h>

# include "FastNoiseLite.h"
#include <thread>
#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <vector>
#include <mutex>
bool  shift = false;
float delta = 0;
float3 moveto = { 0,0,0 };
sutil::Trackball trackball;
sutil::Camera cam;

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;


float3 playerpos = { 0,0,2.0f };


void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit(1);
}


static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}
/*
   */




int viewdist = 6;
const int size = 32;
const int halfsize = size / 2;













float3 latestindex = { 0,0,0 };
class Vertexmanager {       // The class
public:             // Access specifier
    std::vector<float3> vertices;
    std::vector<float3> newvertices;

    std::vector<float4> uvs;
    std::vector<float4> newuvs;

    std::vector<float3> index;
    std::vector<float3> newindex;
    void removechunk(float3 pos);
    void cull();
   void updatechunk(float3 po);
    void updatenew();
    void add(float3 pos,float4 uvs,float3 i);
    void addtri(float3 a, float3 b, float3 c,float4 uva, float2 uvb, float2 uvc,float3 i);
    void addplane(float3 a, float3 b, float3 c, float3 d,float mat,float3 i);
    int getlatestindex();

};



int Vertexmanager::getlatestindex() {
    return vertices.size() + newvertices.size() - 2;

}
class chunk {       // The class
public:             // Access specifier
    void Generate(Vertexmanager& verts, FastNoiseLite Noise);
    void remove();

    float3 position;
    chunk();
    chunk(const chunk& obj);  // copy constructor

};
chunk::chunk() {

}
chunk::chunk(const chunk& obj) {
    //copy constuctor
  //  std::cout << "Copy constructor allocating ptr.";
   // copy the value
}

std::mutex lock;
std::vector<float3> inputs;
;
struct Float3Compare
{
    bool operator() (const float3& lhs, const float3& rhs) const
    {
        return lhs.x < rhs.x
            || (lhs.x == rhs.x && (lhs.y < rhs.y
                || (lhs.y == rhs.y && lhs.z < rhs.z)));
    }
};

class changemanager{       // The class
public:
    // Access specifier
    void registerchange(float4 w);
    std::map<float3, int, Float3Compare> changes;
    float getchange(float3 w);
};

void changemanager::registerchange(float4 w) {

    changes[make_float3(w)] = w.w;

}

float changemanager::getchange(float3 w) {
    if (changes.find(w) == changes.end()) {
        return -1;
    }
    else {
        return changes[w];
    }
}
class world {       // The class
public:  
    // Access specifier
    FastNoiseLite worldnoise;
    changemanager change;
    world() {

        worldnoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    };
    Vertexmanager verts;
    std::vector<float3> chunks;


    void addchunk(float3 pos);
};
world mainworld;

void Vertexmanager::add(float3 pos,float4 uvs,float3 i) {
    lock.lock();
    newvertices.push_back(pos);
    newuvs.push_back(uvs);
    newindex.push_back(i);
    lock.unlock();
}
float distance(float3 a, float3 b)
{
    float d = sqrt(pow(b.x - a.x, 2) +
        pow(b.y - a.y, 2) +
        pow(b.z - a.z, 2) * 1.0);


    return d;
}
float squaredistance(float3 a, float3 b)
{
    float x = abs(a.x - b.x);
    float y = abs(a.y - b.y);
    float z = abs(a.z - b.z);

    return std::max(z, std::max(x, y));
}
bool Within3DManhattanDistance(float3 c1, float3 c2, float distance)
{
    float dx = abs(c2.x - c1.x);
    float dy = abs(c2.y - c1.y);
    float dz = abs(c2.z - c1.z);

    if (dx > distance) return false; // too far in x direction
    if (dy > distance) return false; // too far in y direction
    if (dz > distance) return false; // too far in z direction

    return true; // we're within the cube
}


float3 getpos(float3 where)
{
    where.x = roundf(where.x / size) * size;
    where.y = roundf(where.y / size) * size;
    where.z = roundf(where.z / size) * size;
    return where;

}
float3 cpos;
void Vertexmanager::updatenew() {

    /*
    lock.lock();

    while (newvertices.size() > 0) {


      
        vertices.push_back(newvertices[0]);

        newvertices.erase(newvertices.begin());


        uvs.push_back(newuvs[0]);

        newuvs.erase(newuvs.begin());


        index.push_back(newindex[0]);

        newindex.erase(newindex.begin());
     
    }
       lock.unlock();
*/
    
     

       if (newvertices.size() > 0) {
           lock.lock();
           int i = 0;
           while (i < newvertices.size()) {



               vertices.push_back(newvertices[i]);




               uvs.push_back(newuvs[i]);




               index.push_back(newindex[i]);


               i++;
           }




           newvertices.clear();




           newuvs.clear();




           newindex.clear();
           lock.unlock();
       }

}




void Vertexmanager::cull() {

    cpos = getpos(playerpos);
   
    //check very third and if far away destroy two connect ones
    /*
    int i = 0;
    while(i<vertices.size()/6) {

        if (squaredistance(cpos, vertices[i * 3]) > viewdist * size + 0.1) {


            vertices.erase(vertices.begin()+(i*3));
            vertices.erase(vertices.begin() + (i * 3)+1);
            vertices.erase(vertices.begin() + (i * 3) + 2);
        }
        else {


            i++;
        }
    }
    

    */

    

  /*  mainworld.chunks.erase(std::remove_if(mainworld.chunks.begin(), mainworld.chunks.end() ,
        [](float3 i) {

     
            return !Within3DManhattanDistance(cpos, i, viewdist*size);

             
           
              



        }), mainworld.chunks.end());
        */

    int numberdelc = 0;


    for (float3 chunk : mainworld.chunks) {

        if (!Within3DManhattanDistance(cpos, chunk, viewdist * size)) {
            numberdelc++;

        }



    }


    int numberdel = 0;


  for (float3 value : index) { 
  
      if (!Within3DManhattanDistance(cpos, value, viewdist * size)) {
          numberdel++;

      }
  
  
  
  }
    //check very third and if far away destroy two connect ones



  
            //make vertex number divisble by 3 for coorect traingle splitting

  mainworld.chunks.erase(mainworld.chunks.begin(), mainworld.chunks.begin() + numberdelc);
  index.erase(index.begin(), index.begin() + numberdel);
            vertices.erase(vertices.begin(), vertices.begin() + numberdel);
            uvs.erase(uvs.begin(), uvs.begin() + numberdel);
        
  
}

void Vertexmanager::removechunk(float3 po) {

    float3 pos = getpos(po - make_float3(halfsize, halfsize, halfsize));
    /*

    int i = 0;
    while (i < vertices.size()) {

        if (index[i].x == pos.x && index[i].y ==pos.y && index[i].z ==pos.z) {


           vertices.erase(vertices.begin() +i );
            index.erase(index.begin() + i);
           uvs.erase(uvs.begin() + i);
       
        }
        else {


            i++;
        }
    }

    */
 


    std::vector <float3> vertices2;
    std::vector <float4>uvs2;
    std::vector <float3>index2;

  

    for (int i = 0; i < vertices.size(); i++)
    {
        if (index[i].x == pos.x && index[i].y == pos.y && index[i].z == pos.z) {

           
        }
        else {

            vertices2.push_back(vertices[i]);
            uvs2.push_back(uvs[i]);
            index2.push_back(index[i]);
        }
         
    }

    vertices = vertices2;
    uvs = uvs2;
    index = index2;
    /*

    int x = 0;
    index.erase(std::remove_if(index.begin(), index.end(),
        [pos,&x](float3 i) {

            if (i.x == pos.x && i.x == pos.y && i.x == pos.z) {
            
                return true;

            }
            x++;
            return false;




           


        }), index.end());

        */

            //make vertex number divisble by 3 for coorect traingle splitting
  //  index.erase(index.begin(), index.begin() + numberdel);
  //  vertices.erase(vertices.begin(), vertices.begin() + numberdel);
  //  uvs.erase(uvs.begin(), uvs.begin() + numberdel);


}


void Vertexmanager::updatechunk(float3 pos) {
    lock.lock();
   // inputs.push_back(getpos(pos - make_float3(16, 16, 16)));
    inputs.insert(inputs.begin(), getpos(pos - make_float3(halfsize, halfsize, halfsize)));
    lock.unlock();
    removechunk(pos);

  
}

int getindex(int x, int y, int z) {
    return x + (size + 3) * (y + (size + 3) * z);

}



float getBlock(float nx, float ny, float nz, FastNoiseLite noise) {
    float r = noise.GetNoise((float)nx, (float)ny);
    // float r3d = noise.GetNoise((float)nx*10, (float)ny * 10, (float)nz * 10);
      //    int height = rand() % 10;

    float change = mainworld.change.getchange(make_float3(nx, ny, nz));
    if (change > -1) {
        return change;

    }

    if (nz < r*6) {

      
        return 1;
    





    }
    else {


     
        return 0;



    }


}

float3 raycast(float3 start, float3 dir, float step, int maxrange) {

    dir = normalize(dir);

    int c = 0;

    while (c < maxrange) {

        start += dir * step;
        float3 blockpos = make_float3(round(start.x), round(start.y), round(start.z));

        if (getBlock(blockpos.x, blockpos.y, blockpos.z, mainworld.worldnoise) > 0) {

            return blockpos;
        }

            c++;
    }

    return { 0,0,0 };

}

bool isblock(float3 start) {

 
        float3 blockpos = make_float3(round(start.x), round(start.y), round(start.z));

        if (getBlock(blockpos.x, blockpos.y, blockpos.z, mainworld.worldnoise) > 0) {

            return true;
        }

      

    return false;

}
void chunk::Generate(Vertexmanager& verts,FastNoiseLite noise) {



  
    int* blocks = new int[(size + 3) * (size + 3) * (size + 3)];


    bool empty = true;

    for (int x = 0; x < size + 2; x++)
    {
        for (int y = 0; y < size + 2; y++)
        {
            for (int z = 0; z < size + 2; z++)
            {

                int nx = (int)position.x + x - 1;
                int ny = (int)position.y + y - 1;
                int nz = (int)position.z + z - 1;




                //     srand(nx * ny * nz + nx + ny + nz);

             float val = getBlock(nx, ny, nz, noise);

             blocks[getindex(x, y, z)] = val;
             if (val > 0 && empty == true) {

                 empty = false;
                }
                







            }

        }


    }


    if (empty == false) {

        for (int x = 0; x < size + 1; x++)
        {
            for (int y = 0; y < size + 1; y++)
            {
                for (int z = 0; z < size + 1; z++)
                {

                    int nx = (int)position.x + x;
                    int ny = (int)position.y + y;
                    int nz = (int)position.z + z;

                    float3 pos = make_float3(nx, ny, nz);

                    int mat = 0;

                  
                    //1 for reflective zero for diffuse


                    if (blocks[getindex(x + 1, y + 1, z + 1)] == 0)
                    {

                        if (blocks[getindex(x + 2, y + 1, z + 1)] > 0)
                        {

                            mat = blocks[getindex(x + 2, y + 1, z + 1)] - 1;

                            verts.addplane(make_float3(0.5f, -0.5f, 0.5f) + pos, make_float3(0.5f, -0.5f, -0.5f) + pos, make_float3(0.5f, 0.5f, 0.5f) + pos, make_float3(0.5f, 0.5f, -0.5f) + pos, mat,position);





                        }
                        if (blocks[getindex(x, y + 1, z + 1)] > 0)
                        {
                            mat = blocks[getindex(x, y + 1, z + 1)] - 1;
                            verts.addplane(make_float3(-0.5f, -0.5f, 0.5f) + pos, make_float3(-0.5f, -0.5f, -0.5f) + pos, make_float3(-0.5f, 0.5f, 0.5f) + pos, make_float3(-0.5f, 0.5f, -0.5f) + pos, mat, position);








                        }
                        if (blocks[getindex(x + 1, y, z + 1)] > 0)
                        {

                            mat = blocks[getindex(x + 1, y, z + 1)] - 1;


                            verts.addplane(make_float3(-0.5f, -0.5f, -0.5f) + pos, make_float3(0.5f, -0.5f, -0.5f) + pos, make_float3(-0.5f, -0.5f, 0.5f) + pos, make_float3(0.5f, -0.5f, 0.5f) + pos, mat, position);










                        }
                        if (blocks[getindex(x + 1, y + 2, z + 1)] > 0)
                        {
                            mat = blocks[getindex(x + 1, y + 2, z + 1)] - 1;

                            verts.addplane(make_float3(-0.5f, 0.5f, -0.5f) + pos, make_float3(0.5f, 0.5f, -0.5f) + pos, make_float3(-0.5f, 0.5f, 0.5f) + pos, make_float3(0.5f, 0.5f, 0.5f) + pos, mat,position);






                        }


                        if (blocks[getindex(x + 1, y + 1, z)] > 0)
                        {

                            mat = blocks[getindex(x + 1, y + 1, z)] - 1;
                            verts.addplane(make_float3(-0.5f, -0.5f, -0.5f) + pos, make_float3(0.5f, -0.5f, -0.5f) + pos, make_float3(-0.5f, 0.5f, -0.5f) + pos, make_float3(0.5f, 0.5f, -0.5f) + pos, mat, position);





                        }
                        if (blocks[getindex(x + 1, y + 1, z + 2)] > 0)
                        {

                            mat = blocks[getindex(x + 1, y + 1, z + 2)]-1;
                            verts.addplane(make_float3(-0.5f, -0.5f, 0.5f) + pos, make_float3(0.5f, -0.5f, 0.5f) + pos, make_float3(-0.5f, 0.5f, 0.5f) + pos, make_float3(0.5f, 0.5f, 0.5f) + pos, mat, position);


                        }




                    }


                }

            }


        }
    }




    delete[] blocks;
}

void chunk::remove() {



}


bool canedit = false;
void Vertexmanager::addtri(float3 a, float3 b, float3 c, float4 uva, float2 uvb, float2 uvc,float3 i) {
    add(a,uva,i);
    add(b,make_float4(uvb,0,0),i);
    add(c, make_float4(uvc, 0, 0),i);
}
struct compare
{
    float3 key;
    compare(float3 const& i) : key(i) {}

    bool operator()(float3 const& i) {
        return (i.x == key.x && i.y == key.y && i.z == key.z);
    }
};
void world::addchunk(float3 pos) {



    chunk c;
    c.position = pos;
  
    c.Generate(verts, worldnoise);
    lock.lock();
    chunks.push_back(pos);
    lock.unlock();


    /*    if (std::find_if(chunks.begin(), chunks.end(), compare(pos)) != chunks.end() || distance(playerpos, pos) > size * viewdist) {

        }
        else {

        */
}




bool threadrunning = false;
void Vertexmanager::addplane(float3 a, float3 d, float3 b, float3 c, float mat,float3 i) {
   // addtri(a, b, c, { 0,1,mat,0 }, { 0,0 }, {1,1},i);
  //  addtri(a, d, c, { 1,0,mat,0 }, { 1,1 }, { 0,0 },i);

    float y = 15;

    
    float x = 2;
    if (a.z < -5) {
        x = 0;
    }

    float sx = 32;
    float sy = 16;
    addtri(a, b, c, { x / sx,(1.0f + y) / sy,mat,0 }, { x / sx,y / sy }, { (1.0f + x) / sx,(1.0f + y) / sy }, i);
    addtri(a, d, c, { (1.0f + x) / sx, y / sy, mat, 0 }, { x / sx,y / sy }, { (1.0f + x) / sx,(1.0f + y) / sy }, i);
}
//{ (1.0f + x) / sx, y / sy, mat, 0 }, { (1.0f + x) / sx,(1.0f + y) / sy }, { x / sx,y / sy }

// float2 uv1 = { 0,1 };
 //   float2 uv2 = { 0,0 };
 //    float2 uv3 = { 1,1 };
void chunkthread() {
    threadrunning = true;



    while (threadrunning) {
        if (canedit) {




            int size = inputs.size();
            float3 in;
            bool n = false;
            if (size > 0) {
                lock.lock();
                n = true;
                in = inputs[0];
                inputs.erase(inputs.begin());
                lock.unlock();
            }




            if (n) {

              

                mainworld.addchunk(in);

            }





        }

        /* if (canedit) {
            j++;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (j < 6000) {
                for (int x = 0; x < 5; x++)
                {
                    for (int y = 0; y < 5; y++)
                    {
                        mainworld.addchunk({ j * 5 + 0.0001f + x,y+0.000f,(rand() % 100) / 100.0f });
                    }


                }

            }

           int j = 0;
    float3 now = make_float3(0)+ (size/2);
    float3 nnow =make_float3(0)+(size / 2);
    float3 center = { 0,0,0 };

               if (playerpos.x > now.x)
            {


                now.x += size;
                nnow.x += size;
                center.x += size;
                mainworld.addchunk(center);
            }
            if (playerpos.z > now.z)
            {


                now.z += size;
                nnow.z += size;

                center.z += size;
                mainworld.addchunk(center);
            }
            if (playerpos.x < nnow.x)
            {


                nnow.x -= size;
                now.x -= size;

                center.x -= size;
                mainworld.addchunk(center);
            }
            if (playerpos.z < nnow.z)
            {


                nnow.z -= size;
                now.z -= size;
                center.z -= size;
                mainworld.addchunk(center);
            }

            if (playerpos.y < nnow.y)
            {


                nnow.y -= size;
                now.y -= size;

                center.y -= size;
                mainworld.addchunk(center);
            }


            if (playerpos.y > now.y)
            {


                now.y += size;
                nnow.y += size;
                center.y += size;
                mainworld.addchunk(center);
            }

        }*/


    }
    //   mainworld.startchunks();

}





void displaySubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window)
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}
bool jumping = false;
static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
 
    float3 dir = normalize(playerpos - moveto);
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)

        {
            threadrunning = false;
            glfwSetWindowShouldClose(window, true);
        }
    }
    else if (key == GLFW_KEY_W)
    {

        if (shift) {
            playerpos.x -= dir.x  ;
            playerpos.y -= dir.y ;
        }
        else {
            playerpos.x -= dir.x / 5;
            playerpos.y -= dir.y / 5;

        }

    }
    else if (key == GLFW_KEY_S)
    {
        playerpos.x += dir.x / 10 ;
        playerpos.y += dir.y / 10;
    }
    else if (key == GLFW_KEY_D)
    {
        playerpos.x += 0.1f;
    }
    else if (key == GLFW_KEY_A)
    {
        playerpos.x -= 0.1f;
    }
     if (key == GLFW_KEY_SPACE)
    {
         if (action == GLFW_PRESS) {
             jumping = true;
         }
         if (action == GLFW_RELEASE) {

             jumping = false;
         }
        
        playerpos.z +=0.5;
    }
    else if (key == GLFW_KEY_X)
    {
        playerpos.z -= 0.1f;
    }
    else if (key == GLFW_KEY_LEFT_SHIFT)
    {
        shift = true;
    }
    else if (key == GLFW_KEY_RIGHT_SHIFT)
    {
        shift = false;
    }

}
float radians(float degrees)
{
    return degrees * M_PIf / 180.0f;
}
float degrees(float radians)
{
    return radians * M_1_PIf * 180.0f;
}

bool cnot(float3 a, float3 b) {

    if (a.x == b.x && a.y == b.y && a.z == b.z) {

        return false;
    }
    else {

        return true;
    }

}


static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
 

    if (action == GLFW_PRESS)
    {
      
        float3 cast = raycast(playerpos, cam.direction(), 0.1, 400);
        if (cast.x == 0 && cast.y == 0 && cast.z == 0) {

      //      std::cout << "none \n";


        }
        else {
          //  cast = cast+ make_float3(0, 0, 2);
        //    std::cout << "(" << cast.x << "," << cast.y << "," << cast.z << ") \n";
      //      mainworld.verts.addplane(make_float3(-0.5f, -0.5f, -0.5f) + cast, make_float3(0.5f, -0.5f, -0.5f) + cast, make_float3(-0.5f, 0.5f, -0.5f) + cast, make_float3(0.5f, 0.5f, -0.5f) + cast, 0, make_float3( 0,0,0 ));

            int u = 0;

            if (button == GLFW_MOUSE_BUTTON_RIGHT || button == GLFW_MOUSE_BUTTON_MIDDLE) {

                u = 1;
                if (button == GLFW_MOUSE_BUTTON_MIDDLE) {

                    u = 3;
                }

           
               float3 start = (normalize(playerpos-cast));
               if (abs(start.x) > abs(start.z) && abs(start.x) > abs(start.y)) {

                   start.z = 0;
                   start.y = 0;
               }

               if (abs(start.z) > abs(start.x) && abs(start.z) > abs(start.y)) {

                   start.x = 0;
                   start.y = 0;
               }

               if (abs(start.y) > abs(start.z) && abs(start.y) > abs(start.x)) {

                   start.z = 0;
                   start.x = 0;
               }

               start = make_float3(roundf(start.x), roundf(start.y), roundf(start.z));
              


               cast = cast + start;

            }
            mainworld.change.registerchange(make_float4(cast, u));
            mainworld.verts.updatechunk(cast);
            float3 pos = getpos(cast - make_float3(halfsize, halfsize, halfsize));
            float3 po =cast - make_float3(halfsize, halfsize, halfsize);

       
            if (cnot(getpos(po + make_float3(0, 0, 2)), pos)) {
                mainworld.verts.updatechunk(cast + make_float3(0, 0, 2));

         }

            if (cnot(getpos(po + make_float3(0, 0, -2)), pos)) {
                mainworld.verts.updatechunk(cast + make_float3(0, 0, -2));

            }
       

            if (cnot(getpos(po + make_float3(0, 2, 0)), pos)) {
                mainworld.verts.updatechunk(cast + make_float3(0, 2, 0));

            }

            if (cnot(getpos(po + make_float3(0, -2, 0)), pos)) {
                mainworld.verts.updatechunk(cast + make_float3(0, -2, 0));

            }

            if (cnot(getpos(po + make_float3(2, 0, 0)), pos)) {
                mainworld.verts.updatechunk(cast + make_float3(2, 0, 0));

            }

            if (cnot(getpos(po + make_float3(-2, 0, 0)), pos)) {
                mainworld.verts.updatechunk(cast + make_float3(-2, 0, 0));

            }

         
          

        }
      
    }
    else
    {
      
    }
}


void quechunk(float3 pos) {
    if (std::find_if(mainworld.chunks.begin(), mainworld.chunks.end(), compare(pos)) != mainworld.chunks.end()) {






    }
    else {
        lock.lock();
        inputs.push_back(pos);
        lock.unlock();
    }
}

int main(int argc, char* argv[])
{
 
    std::string outfile;
    int         width = 2000;
    int         height = 1080;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--file" || arg == "-f")
        {
            if (i < argc - 1)
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg.substr(0, 6) == "--dim=")
        {
            const std::string dims_arg = arg.substr(6);
            sutil::parseDimensions(dims_arg.c_str(), width, height);
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try
    {


        char log[2048]; // For error reporting from OptiX creation functions


        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK(cudaFree(0));

            // Initialize the OptiX API, loading all API entry points
            OPTIX_CHECK(optixInit());

            // Specify context options
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &context_log_cb;
            options.logCallbackLevel = 4;

            // Associate a CUDA context (and therefore a specific GPU) with this
            // device context
            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        }


        //
        // accel handling
        //
        OptixTraversableHandle gas_handle;
        CUdeviceptr            d_gas_output_buffer;


        //
        // Create module
        //
        OptixModule module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

            pipeline_compile_options.usesMotionBlur = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues = 2;
            pipeline_compile_options.numAttributeValues = 2;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
            pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

            size_t      inputSize = 0;
            const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "shader.cu", inputSize);
            size_t sizeof_log = sizeof(log);

            OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
                context,
                &module_compile_options,
                &pipeline_compile_options,
                input,
                inputSize,
                log,
                &sizeof_log,
                &module
            ));
        }

        //
        // Create program groups
        //
        OptixProgramGroup raygen_prog_group = nullptr;
        OptixProgramGroup miss_prog_group = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc = {}; //
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            size_t sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &raygen_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &raygen_prog_group
            ));

            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &miss_prog_group
            ));

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &hitgroup_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &hitgroup_prog_group
            ));
        }

        //
        // Link pipeline
        //
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t    max_trace_depth = 2;
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = max_trace_depth;
            pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            size_t sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixPipelineCreate(
                context,
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof(program_groups) / sizeof(program_groups[0]),
                log,
                &sizeof_log,
                &pipeline
            ));

            OptixStackSizes stack_sizes = {};
            for (auto& prog_group : program_groups)
            {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                0,  // maxCCDepth
                0,  // maxDCDEpth
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state, &continuation_stack_size));
            OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state, continuation_stack_size,
                1  // maxTraversableDepth
            ));
        }

        sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
        // Set up shader binding table
        //
        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof(RayGenSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
            RayGenSbtRecord rg_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(raygen_record),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
            ));

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof(MissSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
            MissSbtRecord ms_sbt;





            //bg color
            ms_sbt.data = { 0.3f, 0.3f, 0.3f };




            OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(miss_record),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
            ));

            CUdeviceptr hitgroup_record;
            size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
            HitGroupSbtRecord hg_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(hitgroup_record),
                &hg_sbt,
                hitgroup_record_size,
                cudaMemcpyHostToDevice
            ));

            sbt.raygenRecord = raygen_record;
            sbt.missRecordBase = miss_record;
            sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
            sbt.missRecordCount = 1;
            sbt.hitgroupRecordBase = hitgroup_record;
            sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
            sbt.hitgroupRecordCount = 1;
        }



        //
        // launch
        //
        {



        }

        //
        // Display results
        //
        int i = 0;

        GLFWwindow* window = sutil::initUI("Minecraft 3", width, height);
    
       
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetKeyCallback(window, keyCallback);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        //
        // Render loop
        //


        {


            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                output_buffer_type,
                width,
                height
            );

            CUstream                       stream = 0;
            CUDA_CHECK(cudaStreamCreate(&stream));
       

            output_buffer.setStream(stream);
            sutil::GLDisplay gl_display;

            std::chrono::duration<double> state_update_time(0.0);
            std::chrono::duration<double> render_time(0.0);
            std::chrono::duration<double> display_time(0.0);
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
            const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
            OptixBuildInput triangle_input = {};
            triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;

            triangle_input.triangleArray.flags = triangle_input_flags;
            triangle_input.triangleArray.numSbtRecords = 1;

            std::thread threader(chunkthread);


           //     mainworld.addchunk({ 0,0,0 });



                for (int x = 0; x < viewdist; x++) {
                    for (int y = 0; y < viewdist; y++) {

                        for (int z = 0; z < viewdist; z++) {
                            float3 pos = make_float3(0,0,0) + make_float3(x * size - (viewdist / 2 * size), y * size - (viewdist / 2 * size), z * size - (viewdist / 2 * size));










                            mainworld.addchunk(pos);

                        }


                    }



                }
            

            mainworld.verts.updatenew();
            // Use default options for simplicity.  In a real use case we would want to
            // enable compaction, etc

            // Triangle build input: simple list of three vertices


            size_t vertices_size;
            CUdeviceptr d_vertices = 0;


            OptixAccelBufferSizes gas_buffer_sizes;

            CUdeviceptr d_temp_buffer_gas;



            // We can now free the scratch space buffer used during build and the vertex
            // inputs, since they are not needed by our trivial shading method

            CUdeviceptr d_param;
            Params params;

            CUdeviceptr  d_uvs = 0;



         
          
            auto t0 = std::chrono::steady_clock::now();
            auto t1 = std::chrono::steady_clock::now();
            cam.setLookat({ 10,0,0 });





            cam.setUp({ 0.0f, 0.0f, 3.0f });
            cam.setFovY(45.0f);
            cam.setAspectRatio((float)width / (float)height);

            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            trackball.setCamera(&cam);
            trackball.setMoveSpeed(10.0f);
            trackball.setReferenceFrame(


                make_float3(1.0f, 0.0f, 0.0f),
                make_float3(0.0f, 1.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f)
            );
            trackball.reinitOrientationFromCamera();
            trackball.setGimbalLock(true);
            trackball.setViewMode(sutil::Trackball::EyeFixed);

            trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
            clock_t start = clock();
            clock_t nowt;
            float3 now = getpos(playerpos);














            // Create texture object

            cudaTextureDesc* tex_desc = {};

            params.tex = sutil::loadTexture("C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0/SDK/build/bin/Debug/game.ppm", { 0,0,0 }, tex_desc).texture;





            std::cout << "tex loaded";




            size_t uvs_size;



            //    std::cout << cam.lookat().x;
            do
            {
                nowt = clock();
                delta = nowt - start;

                start = nowt;

                t0 = std::chrono::steady_clock::now();



                //switch from raycastto basic check of blocks below player
                //add physic and jump
            
             

                if (isblock(playerpos + make_float3(0, 0, -2.5))) {

                     playerpos.z += 0.05;

                }
                if (isblock(playerpos + make_float3(0, 0, -2))) {

                    playerpos.z += 0.1;

                }
                if (!jumping) {
                    playerpos.z -= 0.05;





                }




















                canedit = false;


                float3 cposs = getpos(playerpos);



                if (cposs.x != now.x || cposs.y != now.y || cposs.z != now.z)
                {
                    mainworld.verts.cull();
                    lock.lock();
                    inputs.clear();
                    lock.unlock();
                    now = cposs;
                    

                        quechunk(cposs);
                    
                    
                  
                    for (int x = 0; x < viewdist; x++) {
                        for (int y = 0; y < viewdist; y++) {

                            for (int z = 0; z < viewdist; z++) {
                                float3 pos = cposs + make_float3(x * size - (viewdist / 2 * size), y * size - (viewdist / 2 * size), z * size - (viewdist / 2 * size));
                          




                                 

                                
                              

                                    quechunk(pos);
                                
                            }


                        }



                    }


                }
              
                mainworld.verts.updatenew();

                // do stuff here
            

                // mainworld.addchunk({ 0,1,(-i / 1000.01f) });

                 //rebuild

                {
                  
                    vertices_size = sizeof(float3) * mainworld.verts.vertices.size();
                    d_vertices = 0;
                    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
                    CUDA_CHECK(cudaMemcpy(
                        reinterpret_cast<void*>(d_vertices),
                        mainworld.verts.vertices.data(),
                        vertices_size,
                        cudaMemcpyHostToDevice
                    ));



                    uvs_size = sizeof(float4) * mainworld.verts.vertices.size();
                     d_uvs = 0;
                    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_uvs), uvs_size));
                    CUDA_CHECK(cudaMemcpy(
                        reinterpret_cast<void*>(d_uvs),
                        mainworld.verts.uvs.data(),
                        uvs_size,
                        cudaMemcpyHostToDevice
                    ));






                    // Our build input is a simple list of non-indexed triangle vertices
                    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(mainworld.verts.vertices.size());
                    triangle_input.triangleArray.vertexBuffers = &d_vertices;


                    OPTIX_CHECK(optixAccelComputeMemoryUsage(
                        context,
                        &accel_options,
                        &triangle_input,
                        1, // Number of build inputs
                        &gas_buffer_sizes
                    ));

                    CUDA_CHECK(cudaMalloc(
                        reinterpret_cast<void**>(&d_temp_buffer_gas),
                        gas_buffer_sizes.tempSizeInBytes
                    ));
                    CUDA_CHECK(cudaMalloc(
                        reinterpret_cast<void**>(&d_gas_output_buffer),
                        gas_buffer_sizes.outputSizeInBytes
                    ));

                    OPTIX_CHECK(optixAccelBuild(
                        context,
                        0,                  // CUDA stream
                        &accel_options,
                        &triangle_input,
                        1,                  // num build inputs
                        d_temp_buffer_gas,
                        gas_buffer_sizes.tempSizeInBytes,
                        d_gas_output_buffer,
                        gas_buffer_sizes.outputSizeInBytes,
                        &gas_handle,
                        nullptr,            // emitted property list
                        0                   // num emitted properties
                    ));

                    // We can now free the scratch space buffer used during build and the vertex
                    // inputs, since they are not needed by our trivial shading method
                    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
                





                }


                canedit = true;












                i++;

                glfwPollEvents();


                //    mainworld.addchunk({ 0,1,i/1000.01f });












                  //  cam.setLookat({ 0,0,0 });
                    //cam.setDirection(make_float3(radians(xpos/1000), radians(-ypos / 1000), radians(1)));


                t1 = std::chrono::steady_clock::now();
                state_update_time += t1 - t0;
                t0 = t1;
                cam.setEye(playerpos);
                glfwGetCursorPos(window, &xpos, &ypos);

                trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), width, height);


                moveto = cam.lookat();
                params.verts = reinterpret_cast<float3*>(d_vertices);
                params.uvs = reinterpret_cast<float4*>(d_uvs);
                params.image = output_buffer.map();
                params.image_width = width;
                params.image_height = height;
                params.handle = gas_handle;
                params.cam_eye = cam.eye();
                params.playerpos = playerpos;

                cam.UVWFrame(params.cam_u, params.cam_v, params.cam_w);


                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
                CUDA_CHECK(cudaMemcpy(
                    reinterpret_cast<void*>(d_param),
                    &params, sizeof(params),
                    cudaMemcpyHostToDevice
                ));


                OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, /*depth=*/1));
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_uvs)));
                output_buffer.unmap();
                CUDA_SYNC_CHECK();

                t1 = std::chrono::steady_clock::now();
                render_time += t1 - t0;
                t0 = t1;

                displaySubframe(output_buffer, gl_display, window);
                t1 = std::chrono::steady_clock::now();
                display_time += t1 - t0;

               sutil::displayStats(state_update_time, render_time, display_time);
           //    std::cout << mainworld.verts.vertices.size() << "\n";

                glfwSwapBuffers(window);

              
            } while (!glfwWindowShouldClose(window));
            CUDA_SYNC_CHECK();
            threader.join();

        }

        sutil::cleanupUI(window);
        //
        // Cleanup
        //
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));

            OPTIX_CHECK(optixPipelineDestroy(pipeline));
            OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
            OPTIX_CHECK(optixModuleDestroy(module));

            OPTIX_CHECK(optixDeviceContextDestroy(context));
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
