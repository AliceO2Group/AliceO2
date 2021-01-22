set(MACRO_GPU_TARGET "")
if (CUDA_ENABLED)
list(APPEND MACRO_GPU_TARGET O2::DetectorsVertexingCUDA)
endif()
if(HIP_ENABLED)
list(APPEND MACRO_GPU_TARGET O2::DetectorsVertexingHIP)
endif()

o2_add_test_root_macro(runSVertexer.C
                       PUBLIC_LINK_LIBRARIES
                       ${MACRO_GPU_TARGET}
                       LABELS svertexer)