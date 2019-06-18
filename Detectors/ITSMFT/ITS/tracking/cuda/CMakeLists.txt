find_package(cub REQUIRED)

message(STATUS "Building ITS CUDA tracker")

o2_add_library(ITStrackingCUDA
               SOURCES src/Context.cu
                       src/DeviceStoreNV.cu
                       src/Stream.cu
                       src/TrackerTraitsNV.cu
                       src/VertexerTraitsGPU.cu
                       src/Utils.cu
               PUBLIC_LINK_LIBRARIES O2::ITStracking cub::cub
               TARGETVARNAME targetName)

set_property(TARGET ${targetName} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
