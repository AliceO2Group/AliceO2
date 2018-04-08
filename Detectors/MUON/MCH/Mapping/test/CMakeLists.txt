O2_SETUP(NAME MCHMappingTest)
set(BUCKET_NAME mch_mapping_test_bucket)

# trying to get only one test exe for this module
O2_GENERATE_EXECUTABLE(
        EXE_NAME test_MCHMappingTest
        SOURCES src/Segmentation.cxx src/SegmentationLong.cxx 
        BUCKET_NAME mch_mapping_test_bucket
)

target_link_libraries(test_MCHMappingTest Boost::unit_test_framework)
add_test(NAME test_MCHMappingTest COMMAND test_MCHMappingTest)

if (benchmark_FOUND)
    O2_GENERATE_EXECUTABLE(
            EXE_NAME mch-mapping-bench-segmentation3
            SOURCES src/BenchSegmentation.cxx 
            BUCKET_NAME mch_mapping_test_bucket)
endif ()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/test_random_pos.json
DESTINATION ${CMAKE_CURRENT_BINARY_DIR})