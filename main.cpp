

//glfw
#include <glad/glad.h> 
#include <GLFW/glfw3.h>
//optix
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>


//cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//utils
#include <sutil/sutil.h>
#include <sutil/GLDisplay.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Trackball.h>


//types
#include <map>
#include <array>
#include <vector>
#include <string>


//io
#include <iomanip>
#include <iostream>

//buffer
#include "buffer.h"


//multithreading
#include <thread>
#include <mutex>


//algorithms
# include "FastNoiseLite.h"
#include <algorithm>
#include <atomic>


int viewdist = 6;
const int size = 32;

int         width = 2000;
int         height = 1080;
int MSAA = 2;




const int halfsize = size / 2;

sutil::Camera cam;
std::mutex lock;


//sbt record
template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

//utitlity functions:



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
struct Float3Compare
{
    bool operator() (const float3& lhs, const float3& rhs) const
    {
        return lhs.x < rhs.x
            || (lhs.x == rhs.x && (lhs.y < rhs.y
                || (lhs.y == rhs.y && lhs.z < rhs.z)));
    }
};

struct compare
{
    float3 key;
    compare(float3 const& i) : key(i) {}

    bool operator()(float3 const& i) {
        return (i.x == key.x && i.y == key.y && i.z == key.z);
    }
};
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

int getindex(int x, int y, int z) {
    return x + (size + 3) * (y + (size + 3) * z);

}

//classes
//manager class
class Vertexmanager {     
public:           
    std::vector<float3> vertices;
    std::vector<float3> newvertices;

    std::vector<float4> uvs;
    std::vector<float4> newuvs;

    std::vector<float3> index;
    std::vector<float3> newindex;
    void removechunk(float3 pos);
    void cull(float3 pos, std::vector <float3> &chunks);
  
    void updatenew();
    void add(float3 pos, float4 uvs, float3 i);
    void addtri(float3 a, float3 b, float3 c, float4 uva, float2 uvb, float2 uvc, float3 i);
    void addplane(float3 a, float3 b, float3 c, float3 d, float mat, float3 i, int side);


};



void Vertexmanager::add(float3 pos, float4 uvs, float3 i) {
    lock.lock();
    newvertices.push_back(pos);
    newuvs.push_back(uvs);
    newindex.push_back(i);
    lock.unlock();
}

void Vertexmanager::addtri(float3 a, float3 b, float3 c, float4 uva, float2 uvb, float2 uvc, float3 i) {
    add(a, uva, i);
    add(b, make_float4(uvb, 0, 0), i);
    add(c, make_float4(uvc, 0, 0), i);
}
void Vertexmanager::addplane(float3 a, float3 d, float3 b, float3 c, float mat, float3 i, int side) {

    float y = 15;


    float x = 2;
    if (side < 5) {
        x = 3;

    }
    else if (side == 5) {

        x = 18;

        y = 14;
    }

    if (mat==4) {
        x = 0;
    }

    if (mat == 5) {
        x = 18;

        y = 14;
    }

    //2
    if (mat == 1) {
        x = 15;
        y = 12;
    }
    if (mat == 6) {

        x = 24;

        y = 11;
    }


    if (mat == 7) {

        x = 21;

        y = 14;
    }
    float sx = 32;
    float sy = 16;
    addtri(a, b, c, { x / sx,(1.0f + y) / sy,mat,0 }, { x / sx,y / sy }, { (1.0f + x) / sx,(1.0f + y) / sy }, i);
    addtri(a, d, c, { (1.0f + x) / sx, y / sy, mat, 0 }, { x / sx,y / sy }, { (1.0f + x) / sx,(1.0f + y) / sy }, i);
}
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




void Vertexmanager::cull(float3 pos, std::vector <float3>& chunks) {

    pos = getpos(pos);

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


    int numberdelc = 0;


    for (float3 chunk : chunks) {

        if (!Within3DManhattanDistance(pos, chunk, viewdist * size)) {
            numberdelc++;

        }



    }


    int numberdel = 0;


    for (float3 value : index) {

        if (!Within3DManhattanDistance(pos, value, viewdist * size)) {
            numberdel++;

        }



    }
   
   chunks.erase(chunks.begin(),chunks.begin() + numberdelc);
    index.erase(index.begin(), index.begin() + numberdel);
    vertices.erase(vertices.begin(), vertices.begin() + numberdel);
    uvs.erase(uvs.begin(), uvs.begin() + numberdel);


}

void Vertexmanager::removechunk(float3 po) {

    float3 pos = getpos(po - make_float3(halfsize, halfsize, halfsize));
  


    std::vector <float3> vertices2;
    std::vector <float4>uvs2;
    std::vector <float3>index2;


    int isze = vertices.size();
    vertices2.reserve(isze);
    uvs2.reserve(isze);
    index2.reserve(isze);
    for (int i = 0; i < isze; i++)
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
 


}



class changemanager {
public:

    void registerchange(float4 w);
    std::map<float3, int, Float3Compare> changes;
    float getchange(float3 w);
};





void changemanager::registerchange(float4 w) {

    changes[make_float3(w)] = w.w;

}

float changemanager::getchange(float3 w) {
    if (changes.count(w) == 0) {
        return -1;
    }
    else {
        return changes[w];
    }
}





//player class
class Player{
    public:
        float3 moveto = { 0,0,0 };
        float3 playerpos = { 0,0,2.0f };
        float vz = 0;

        void update(float delta);
    };


void Player::update(float delta){

   playerpos.z += vz*delta;
    }
//chunk class
class chunk {      
public:          
    void Generate(Vertexmanager& verts);
  

    float3 position;
  //  chunk();
 //   chunk(const chunk& obj);  copy constructor

};




//global wolrd class
class world {      
public:
  
    FastNoiseLite worldnoise;
    changemanager change;
    world() {

        worldnoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    };
    Vertexmanager verts;
    std::vector<float3> chunks;

    std::vector<float3> inputs;
    Player player;
    float getBlock(float nx, float ny, float nz);
    void addchunk(float3 pos);
    float3 raycast(float3 start, float3 dir, float step, int maxrange);
    bool isblock(float3 start);
    void quechunk(float3 pos );
    void updatechunk(float3 pos);
};

world mainworld;


void chunk::Generate(Vertexmanager& verts) {




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

                float val = mainworld.getBlock(nx, ny, nz);

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


                    int glass = 7;
                    bool isglass =false;
                    if (blocks[getindex(x + 1, y + 1, z + 1)] == glass) {

                        isglass = true;
                    }

                    if (blocks[getindex(x + 1, y + 1, z + 1)] == 0 || isglass )
                    {

                        if (blocks[getindex(x + 2, y + 1, z + 1)] > 0 )
                        {

                           

                            mat = blocks[getindex(x + 2, y + 1, z + 1)] - 1;

                            if (isglass && mat + 1 == glass) {


                            }
                            else {



                                verts.addplane(make_float3(0.5f, -0.5f, 0.5f) + pos, make_float3(0.5f, -0.5f, -0.5f) + pos, make_float3(0.5f, 0.5f, 0.5f) + pos, make_float3(0.5f, 0.5f, -0.5f) + pos, mat, position, 1);

                            }



                        }
                        if (blocks[getindex(x, y + 1, z + 1)] > 0)
                        {
                            mat = blocks[getindex(x, y + 1, z + 1)] - 1;
                            if (isglass && mat + 1 == glass) {


                            }
                            else {
                                verts.addplane(make_float3(-0.5f, -0.5f, 0.5f) + pos, make_float3(-0.5f, -0.5f, -0.5f) + pos, make_float3(-0.5f, 0.5f, 0.5f) + pos, make_float3(-0.5f, 0.5f, -0.5f) + pos, mat, position, 2);


                            }





                        }
                        if (blocks[getindex(x + 1, y, z + 1)] > 0)
                        {

                            mat = blocks[getindex(x + 1, y, z + 1)] - 1;
                            if (isglass && mat + 1 == glass) {


                            }
                            else {

                                verts.addplane(make_float3(0.5f, -0.5f, 0.5f) + pos, make_float3(0.5f, -0.5f, -0.5f) + pos, make_float3(-0.5f, -0.5f, 0.5f) + pos, make_float3(-0.5f, -0.5f, -0.5f) + pos, mat, position, 4);


                            }
                         






                        }
                        if (blocks[getindex(x + 1, y + 2, z + 1)] > 0)
                        {
                            mat = blocks[getindex(x + 1, y + 2, z + 1)] - 1;
                            if (isglass && mat + 1 == glass) {


                            }
                            else {
                                verts.addplane(make_float3(0.5f, 0.5f, 0.5f) + pos, make_float3(0.5f, 0.5f, -0.5f) + pos, make_float3(-0.5f, 0.5f, 0.5f) + pos, make_float3(-0.5f, 0.5f, -0.5f) + pos, mat, position, 3);



                            }


                        }


                        if (blocks[getindex(x + 1, y + 1, z)] > 0)
                        {

                            mat = blocks[getindex(x + 1, y + 1, z)] - 1;
                            if (isglass && mat + 1 == glass) {


                            }
                            else {
                                verts.addplane(make_float3(-0.5f, -0.5f, -0.5f) + pos, make_float3(0.5f, -0.5f, -0.5f) + pos, make_float3(-0.5f, 0.5f, -0.5f) + pos, make_float3(0.5f, 0.5f, -0.5f) + pos, mat, position, 6);


                            }


                        }
                        if (blocks[getindex(x + 1, y + 1, z + 2)] > 0)
                        {

                            mat = blocks[getindex(x + 1, y + 1, z + 2)] - 1;
                            if (isglass && mat + 1 == glass) {


                            }
                            else {
                                verts.addplane(make_float3(-0.5f, -0.5f, 0.5f) + pos, make_float3(0.5f, -0.5f, 0.5f) + pos, make_float3(-0.5f, 0.5f, 0.5f) + pos, make_float3(0.5f, 0.5f, 0.5f) + pos, mat, position, 5);

                            }
                        }




                    }


                }

            }


        }
    }




    delete[] blocks;
}


void world::updatechunk(float3 pos) {
    lock.lock();
    // inputs.push_back(getpos(pos - make_float3(16, 16, 16)));
    inputs.insert(inputs.begin(), getpos(pos - make_float3(halfsize, halfsize, halfsize)));
    lock.unlock();
    verts.removechunk(pos);


}
void world::quechunk(float3 pos ) {
    if (std::find_if(chunks.begin(), chunks.end(), compare(pos)) != chunks.end()) {






    }
    else {
        lock.lock();
        inputs.push_back(pos);
        lock.unlock();
    }
}


void world::addchunk(float3 pos) {



    chunk c;
    c.position = pos;

    c.Generate(verts);
    lock.lock();
    chunks.push_back(pos);
    lock.unlock();


    /*    if (std::find_if(chunks.begin(), chunks.end(), compare(pos)) != chunks.end() || distance(playerpos, pos) > size * viewdist) {

        }
        else {

        */
}


bool world::isblock(float3 start) {


    float3 blockpos = make_float3(round(start.x), round(start.y), round(start.z));

    if (getBlock(blockpos.x, blockpos.y, blockpos.z ) > 0) {

        return true;
    }



    return false;

}

float world::getBlock(float nx, float ny, float nz) {
    float r = worldnoise.GetNoise((float)nx, (float)ny);
    // float r3d = noise.GetNoise((float)nx*10, (float)ny * 10, (float)nz * 10);
      //    int height = rand() % 10;

    float changes = change.getchange(make_float3(nx, ny, nz));
    if (changes > -1) {
        return changes;

    }

    if (nz < r * 6) {

        if (nz < r * 6 - 1) {
            if (nz < r * 6 - 10) {
                return 5;

            }
            return 6;

        }
     
        return 1;






    }
    else {



        return 0;



    }


}

float3 world::raycast(float3 start, float3 dir, float step, int maxrange) {

    dir = normalize(dir);

    int c = 0;

    while (c < maxrange) {

        start += dir * step;
        float3 blockpos = make_float3(round(start.x), round(start.y), round(start.z));

        if (getBlock(blockpos.x, blockpos.y, blockpos.z) > 0) {

            return blockpos;
        }

        c++;
    }

    return { 0,0,0 };

}

//log
static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}



//globals


















bool canedit = false;
bool threadrunning = false;


void chunkthread() {
    threadrunning = true;



    while (threadrunning) {
        if (canedit) {




            int size = mainworld.inputs.size();
            float3 in;
            bool n = false;
            if (size > 0) {
                lock.lock();
                n = true;
                in = mainworld.inputs[0];
                mainworld.inputs.erase(mainworld.inputs.begin());
                lock.unlock();
            }




            if (n) {

              

                mainworld.addchunk(in);

            }





        }

  


    }
  

}







bool shift = false;

static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    
  
   
    if (action == GLFW_RELEASE)
    {
       

    }
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)

        {
            threadrunning = false;
            glfwSetWindowShouldClose(window, true);
        }
        if (key == GLFW_KEY_SPACE)
        {
            mainworld.player.vz += 0.01;


        }
        else if (key == GLFW_KEY_LEFT_SHIFT)
        {
            shift = true;
        }

        else if (key == GLFW_KEY_RIGHT_SHIFT)
        {
            shift = false;
        }
        else if (key == GLFW_KEY_RIGHT)
        {
            MSAA += 1;
            std::cout << "samples set to: " << MSAA << " \n";
        }

        else if (key == GLFW_KEY_LEFT)
        {
            MSAA -= 1;

            std::cout << "samples set to: " << MSAA << " \n";
        }
        else if (key == GLFW_KEY_UP)
        {
            MSAA += 10;
            std::cout << "samples set to: " << MSAA << " \n";
        }

        else if (key == GLFW_KEY_DOWN)
        {
            MSAA -= 10;
            std::cout << "samples set to: " << MSAA << " \n";
        }
        else if (key == GLFW_KEY_0)
        {
            MSAA = 2;
            std::cout << "samples reset to: " << MSAA << " \n";
        }
        else if (key == GLFW_KEY_1)
        {
            MSAA = 50;
            std::cout << "samples set to: " << MSAA << " \n";
        }
        else if (key == GLFW_KEY_2)
        {
            MSAA = 100;
            std::cout << "samples set to: " << MSAA << " \n";
        }

      

    }

  
 
  

}

int selectedblock = 2;

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
 

    if (action == GLFW_PRESS )
    {
        if (button == GLFW_MOUSE_BUTTON_RIGHT || button == GLFW_MOUSE_BUTTON_LEFT) {
            float3 cast = mainworld.raycast(mainworld.player.playerpos, cam.direction(), 0.3, 50);
            if (cast.x == 0 && cast.y == 0 && cast.z == 0) {

                //      std::cout << "none \n";


            }
            else {
                //  cast = cast+ make_float3(0, 0, 2);
              //    std::cout << "(" << cast.x << "," << cast.y << "," << cast.z << ") \n";
            //      mainworld.verts.addplane(make_float3(-0.5f, -0.5f, -0.5f) + cast, make_float3(0.5f, -0.5f, -0.5f) + cast, make_float3(-0.5f, 0.5f, -0.5f) + cast, make_float3(0.5f, 0.5f, -0.5f) + cast, 0, make_float3( 0,0,0 ));

                int u = 0;
             
                if (button == GLFW_MOUSE_BUTTON_RIGHT) {

                    u = selectedblock;




                    float3 start = (normalize(mainworld.player.playerpos - cast));
                    if (abs(start.x) > abs(start.z) && abs(start.x) > abs(start.y)) {

                        start.z = 0;
                        start.y = 0;
                    }



                    else if (abs(start.y) > abs(start.z) && abs(start.y) > abs(start.x)) {

                        start.z = 0;
                        start.x = 0;
                    }


                    else {

                        start.x = 0;
                        start.y = 0;
                    }



                    start = make_float3(roundf(start.x), roundf(start.y), roundf(start.z));



                    float3 newcast = cast + start;


                    if (mainworld.isblock(newcast)) {

                        cast = cast + make_float3(0, 0, 1);

                        if (mainworld.isblock(cast)) {
                            return;

                        }
                    }
                    else {
                        cast = newcast;
                    }

                }





                mainworld.change.registerchange(make_float4(cast, u));
                mainworld.updatechunk(cast);
                float3 pos = getpos(cast - make_float3(halfsize, halfsize, halfsize));
                float3 po = cast - make_float3(halfsize, halfsize, halfsize);


                if (cnot(getpos(po + make_float3(0, 0, 2)), pos)) {
                    mainworld.updatechunk(cast + make_float3(0, 0, 2));

                }

                if (cnot(getpos(po + make_float3(0, 0, -2)), pos)) {
                    mainworld.updatechunk(cast + make_float3(0, 0, -2));

                }


                if (cnot(getpos(po + make_float3(0, 2, 0)), pos)) {
                    mainworld.updatechunk(cast + make_float3(0, 2, 0));

                }

                if (cnot(getpos(po + make_float3(0, -2, 0)), pos)) {
                    mainworld.updatechunk(cast + make_float3(0, -2, 0));

                }

                if (cnot(getpos(po + make_float3(2, 0, 0)), pos)) {
                    mainworld.updatechunk(cast + make_float3(2, 0, 0));

                }

                if (cnot(getpos(po + make_float3(-2, 0, 0)), pos)) {
                    mainworld.updatechunk(cast + make_float3(-2, 0, 0));

                }




            }
        }
        if (button == GLFW_MOUSE_BUTTON_4) {

            selectedblock -= 1;
            std::cout << "Selected block: " << selectedblock << " \n";
        }

        if (button == GLFW_MOUSE_BUTTON_5) {

            selectedblock += 1;
            std::cout << "Selected block: " << selectedblock << " \n";
        }
      
    }
    else
    {
      
    }
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

int main(int argc, char* argv[])
{
 
  
 


   
   
  
    sutil::Trackball trackball;
 
  

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



        
     
      

        GLFWwindow* window = sutil::initUI("OptixCraft", width, height);
    
       
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetKeyCallback(window, keyCallback);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
      


        

        //set up output
            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                output_buffer_type,
                width,
                height
            );

            CUstream                       stream = 0;
            CUDA_CHECK(cudaStreamCreate(&stream));
       

            output_buffer.setStream(stream);
            sutil::GLDisplay gl_display;

            OptixTraversableHandle gas_handle;
                CUdeviceptr            d_gas_output_buffer;


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




                for (int x = 0; x < viewdist; x++) {
                    for (int y = 0; y < viewdist; y++) {

                        for (int z = 0; z < viewdist; z++) {
                            float3 pos = make_float3(0,0,0) + make_float3(x * size - (viewdist / 2 * size), y * size - (viewdist / 2 * size), z * size - (viewdist / 2 * size));










                            mainworld.addchunk(pos);

                        }


                    }



                }
            

            mainworld.verts.updatenew();
       





        



         
          
            auto t0 = std::chrono::steady_clock::now();
            auto t1 = std::chrono::steady_clock::now();
            std::chrono::duration<double> state_update_time(0.0);
            std::chrono::duration<double> render_time(0.0);
            std::chrono::duration<double> display_time(0.0);
            clock_t start = clock();
            clock_t nowt;
            float delta = 0;



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



           
            size_t vertices_size;
            size_t uvs_size;
            CUdeviceptr d_vertices = 0;
            OptixAccelBufferSizes gas_buffer_sizes;
            CUdeviceptr d_temp_buffer_gas;
            CUdeviceptr d_param;
            Params params;
            CUdeviceptr  d_uvs = 0;


            // Create texture object

            cudaTextureDesc* tex_desc = {};


       

            params.tex = sutil::loadTexture("C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0/SDK/build/bin/Debug/game.ppm", { 0,0,0 }, tex_desc).texture;

            std::cout << "tex loaded";





        






        



          


         
            float3 now = getpos(mainworld.player.playerpos);
            int oldsize = 0;
            bool first = true;
            do
            {

                //update clocks
                nowt = clock();
                delta = nowt - start;
                start = nowt;
                t0 = std::chrono::steady_clock::now();


                


                //update player
                //switch from raycastto basic check of blocks below player
                //add physic and jump
         
                mainworld.player.update(delta);
             
                    


                if (glfwGetKey(window, GLFW_KEY_W))
                {
                    float3 dir = normalize(mainworld.player.playerpos - mainworld.player.moveto);

                    if (shift) {
                        mainworld.player.playerpos.x -= dir.x/40 * delta;
                        mainworld.player.playerpos.y -= dir.y/40 * delta;
                    }
                    else {
                        mainworld.player.playerpos.x -= dir.x / 100 * delta;
                        mainworld.player.playerpos.y -= dir.y / 100 * delta;

                    }

                }
                if (glfwGetKey(window, GLFW_KEY_S))
                {

                    float3 dir = normalize(mainworld.player.playerpos - mainworld.player.moveto);
                    mainworld.player.playerpos.x += dir.x / 100 * delta;
                    mainworld.player.playerpos.y += dir.y / 100 * delta;
                }




                
                if (mainworld.isblock(mainworld.player.playerpos + make_float3(0, 0, -2.1))) {

                    mainworld.player.vz = 0;

                }
                else {

                    mainworld.player.vz += -0.0001;
                }
                if (mainworld.isblock(mainworld.player.playerpos + make_float3(0, 0, -2))) {

                    mainworld.player.playerpos.z += 0.1;

                }
               



















                //update world
                canedit = false;


                float3 cposs = getpos(mainworld.player.playerpos);



                if (cposs.x != now.x || cposs.y != now.y || cposs.z != now.z)
                {
                    mainworld.verts.cull(mainworld.player.playerpos,mainworld.chunks);
                    lock.lock();
                    mainworld.inputs.clear();
                    lock.unlock();
                    now = cposs;
                    

                        mainworld.quechunk(cposs);
                    
                    
                  
                    for (int x = 0; x < viewdist; x++) {
                        for (int y = 0; y < viewdist; y++) {

                            for (int z = 0; z < viewdist; z++) {
                                float3 pos = cposs + make_float3(x * size - (viewdist / 2 * size), y * size - (viewdist / 2 * size), z * size - (viewdist / 2 * size));
                          




                                 

                                
                              

                                    mainworld.quechunk(pos);
                                
                            }


                        }



                    }


                }
              
                mainworld.verts.updatenew();

             






                 //rebuild
                
                if(true){
                   if(first == false) {
                        
                       
                    }
                   first == true;

                   
                   oldsize = mainworld.verts.vertices.size();


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


                //mouse
                cam.setEye(mainworld.player.playerpos);
                glfwPollEvents();
                glfwGetCursorPos(window, &xpos, &ypos);
                trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), width, height);
                mainworld.player.moveto = cam.lookat();
                cam.UVWFrame(params.cam_u, params.cam_v, params.cam_w);





            

              

                //clock
                t1 = std::chrono::steady_clock::now();
                state_update_time += t1 - t0;
                t0 = t1;
               

         


              //update params
                params.verts = reinterpret_cast<float3*>(d_vertices);
                params.uvs = reinterpret_cast<float4*>(d_uvs);
                params.image = output_buffer.map();
                params.image_width = width;
                params.image_height = height;

                params.samples = MSAA;
                params.handle = gas_handle;
                params.cam_eye = cam.eye();
                params.playerpos = mainworld.player.playerpos;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
                CUDA_CHECK(cudaMemcpy(
                    reinterpret_cast<void*>(d_param),
                    &params, sizeof(params),
                    cudaMemcpyHostToDevice
                ));

                //launch
                OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, /*depth=*/1));



                //free
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_uvs)));
                output_buffer.unmap();
                CUDA_SYNC_CHECK();



                //final time
                t1 = std::chrono::steady_clock::now();
                render_time += t1 - t0;
                t0 = t1;

                displaySubframe(output_buffer, gl_display, window);
              

                t1 = std::chrono::steady_clock::now();
                display_time += t1 - t0;
               sutil::displayStats(state_update_time, render_time, display_time);
               glfwSwapBuffers(window);
           //    std::cout << mainworld.verts.vertices.size() << "\n";
               /*
               saving screenshots, kind of useless wit ppm format
               if (false) {

                   sutil::ImageBuffer buffer;
                   buffer.data = output_buffer.getHostPointer();
                   buffer.width = output_buffer.width();
                   buffer.height = output_buffer.height();
                   buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

                   sutil::saveImage("C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0/SDK/build/bin/Debug/screenshot.ppm", buffer, false);

                   first = false;
               }
            */


              
            } while (!glfwWindowShouldClose(window));
            CUDA_SYNC_CHECK();
            threader.join();

        

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
