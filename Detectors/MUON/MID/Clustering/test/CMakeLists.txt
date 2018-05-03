O2_SETUP(NAME MIDClusteringTest)
set(BUCKET_NAME mid_clustering_test_bucket)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS testClusterizer.cxx
)

if (benchmark_FOUND)
    O2_GENERATE_EXECUTABLE(
            EXE_NAME benchClusterizer
            SOURCES bench_Clusterizer.cxx
            BUCKET_NAME ${BUCKET_NAME})
endif ()
