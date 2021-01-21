o2_add_test_root_macro(runSVertexer.C
                       PUBLIC_LINK_LIBRARIES
                       $<$<BOOL:${ENABLE_CUDA}>:O2::DetectorsVertexingCUDA>
                       $<$<BOOL:${ENABLE_HIP}>:O2::DetectorsVertexingHIP>
                       LABELS svertexer)