o2_add_test_root_macro(runGPUbenchmark.C
                       PUBLIC_LINK_LIBRARIES O2::HIPbenchmark
                                             O2::GPUCommon
                       LABELS gpu COMPILE_ONLY)