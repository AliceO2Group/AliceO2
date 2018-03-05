O2_SETUP(NAME MCHMappingTest)
set(BUCKET_NAME mch_mapping_test_bucket)

O2_GENERATE_TESTS(
        BUCKET_NAME mch_mapping_test_bucket
        TEST_SRCS src/Segmentation.cxx
)

if (benchmark_FOUND)
    O2_GENERATE_EXECUTABLE(
            EXE_NAME mch-mapping-bench-segmentation3
            SOURCES src/bench_Segmentation.cxx
            BUCKET_NAME mch_mapping_test_bucket)
endif ()
