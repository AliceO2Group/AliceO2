set(MODULE_NAME "ITStrackingCUDA")
O2_SETUP(NAME ${MODULE_NAME})

message(STATUS "Building ITS CUDA tracker")

set(NO_DICT_SRCS
  src/Context.cu
  src/DeviceStoreNV.cu
  src/Stream.cu
  src/TrackerTraitsNV.cu
  src/VertexerTraitsGPU.cu
  src/Utils.cu
)

string(REPLACE ".cu" ".h" NO_DICT_HEADERS "${NO_DICT_SRCS}")
string(REPLACE "src" "include/${MODULE_NAME}" NO_DICT_HEADERS "${NO_DICT_HEADERS}")


set(NO_DICT_HEADERS ${NO_DICT_HEADERS}
    include/${MODULE_NAME}/Array.h
    include/${MODULE_NAME}/PrimaryVertexContextNV.h
    include/${MODULE_NAME}/UniquePointer.h
    include/${MODULE_NAME}/Vector.h
)
set(NO_DICT_SRCS ${NO_DICT_SRCS}
    src/dummy.cxx
)

Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME its_tracking_CUDA_bucket)
O2_GENERATE_LIBRARY()

set_property(TARGET "O2${MODULE_NAME}" PROPERTY CUDA_SEPARABLE_COMPILATION ON)
