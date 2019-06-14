O2_SETUP(NAME "MIDBaseTest")
set(BUCKET_NAME mid_base_test_bucket)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS src/testMapping.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS src/Positions.cxx
)

# target_link_libraries(test_MIDpositions Boost::unit_test_framework MIDBase)
# add_test(NAME test_MIDpositions COMMAND test_MIDpositions)

file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/data/test_random_pos.json
     DESTINATION
     ${CMAKE_CURRENT_BINARY_DIR})

if (benchmark_FOUND)
    O2_GENERATE_EXECUTABLE(
            EXE_NAME benchVector
            SOURCES src/benchVector.cxx
            BUCKET_NAME ${BUCKET_NAME})
endif ()