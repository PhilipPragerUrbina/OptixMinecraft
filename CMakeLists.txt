
OPTIX_add_sample_executable( minecraft3 target_name
  shader.cu
  main.cpp
  buffer.h
  FastNoiseLite.h
  ogt_vox.h
  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )

