o2_setup(NAME MCHMappingTest)
set(BUCKET_NAME mch_mapping_test_bucket)

# trying to get only one test exe for this module
o2_generate_executable(EXE_NAME
                       test_MCHMappingTest
                       SOURCES
                       src/CathodeSegmentation.cxx
                       src/CathodeSegmentationLong.cxx
                       src/Segmentation.cxx
                       BUCKET_NAME
                       mch_mapping_test_bucket)

target_link_libraries(test_MCHMappingTest Boost::unit_test_framework)
add_test(NAME test_MCHMappingTest COMMAND test_MCHMappingTest)

if(benchmark_FOUND)
  o2_generate_executable(EXE_NAME
                         mch-mapping-bench-segmentation3
                         SOURCES
                         src/BenchCathodeSegmentation.cxx
                         src/BenchSegmentation.cxx
                         BUCKET_NAME
                         mch_mapping_test_bucket)
endif()

file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/data/test_random_pos.json
     DESTINATION
     ${CMAKE_CURRENT_BINARY_DIR})
