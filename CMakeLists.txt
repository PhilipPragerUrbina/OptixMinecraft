
OPTIX_add_sample_executable( minecraft3 target_name
  shader.cu
  main.cpp
  buffer.h
  FastNoiseLite.h
  Player.h
  Player.cpp
  changemanager.h
  changemanager.cpp
  ogt_vox.h
  utils.h
  VertexManager.h
  VertexManager.cpp
  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )

