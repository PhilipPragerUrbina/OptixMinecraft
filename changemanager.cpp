#include "changemanager.h"


void changemanager::load_model(const ogt_vox_model* model)
{

    uint32_t voxel_index = 0;
    for (uint32_t z = 0; z < model->size_z; z++) {
        for (uint32_t y = 0; y < model->size_y; y++) {
            for (uint32_t x = 0; x < model->size_x; x++, voxel_index++) {
                // if color index == 0, this voxel is empty, otherwise it is solid.
                uint32_t color_index = model->voxel_data[voxel_index];
                bool is_voxel_solid = (color_index != 0);

                if (is_voxel_solid) {
                    int index = 2;

                    if (color_index == 59) {

                        index = 5;
                    }
                    registerchange(make_float4(x, y, z, index));

                }
                else {


                }
            }
        }
    }

}


void changemanager::registerchange(float4 w) {

    changes[make_float3(w)] = w.w;

}

float changemanager::getchange(float3 w) {
    if (changes.count(w) == 0) {
        return -69;
    }
    else {
        return changes[w];
    }
}