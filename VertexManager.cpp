
#include "VertexManager.h"

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

    if (mat == -2) {

        x = 10;

        y = 11;
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

    pos = getpos(pos,size);

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

    float3 pos = getpos(po - make_float3(halfsize, halfsize, halfsize), size);

    /*
    int start = 0;
    int end = 10;




    bool in = false;
    int e = 0;

    bool done = false;
    for (float3 value : index) {
        if (done == false) {






            if (value.x == pos.x && value.y == pos.y && value.z == pos.z) {
                if (in == false) {

                    in = true;
                    start = e;
                    end = e + 3;

                }
                else {


                  
                }

            }
            else {
                if (in == true) {

                    end = e;

                    in = false;
                    done = true;



                }


            }

            e++;

        }

    }

        index.erase(index.begin() + start, index.begin() + end);
        vertices.erase(vertices.begin() + start, vertices.begin() + end);
        uvs.erase(uvs.begin() + start, uvs.begin() + end);

    std::cout << start << "   " << end << "\n";



    */







    std::vector <float3> vertices2;
    std::vector <float4>uvs2;
    std::vector <float3>index2;


    int isze = vertices.size();
    vertices2 = vertices;
    uvs2 = uvs;
    index2 = index;
    int start = 0;
    int end = 10;




    bool in = false;
    int e = 0;

    bool done = false;
    for (float3 value : index2) {
        if (done == false) {






            if (value.x == pos.x && value.y == pos.y && value.z == pos.z) {
                if (in == false) {

                    in = true;
                    start = e;


                }


                end = e + 1;



            }
            else {
                if (in == true) {



                    in = false;
                    done = true;



                }


            }

            e++;

        }

    }


   
    int num = abs(start - end);

  



    if (num > 0) {
        if (num % 3 != 0) {

            end--;
            num--;
            if (num % 3 != 0) {

                end--;
                num--;
            }

        }
        index2.erase(index2.begin() + start, index2.begin() + end);
        vertices2.erase(vertices2.begin() + start, vertices2.begin() + end);
        uvs2.erase(uvs2.begin() + start, uvs2.begin() + end);
       // Sleep(10);

        vertices = vertices2;
        uvs = uvs2;
        index = index2;
    }






    /*

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


        */





        

    





}





