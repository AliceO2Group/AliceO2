set(MODULE_NAME "MCHPreClustering")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/PreClusterFinder.cxx
)

Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mch_preclustering_bucket)

O2_GENERATE_LIBRARY()

# todo we repeat ourselves because the above macro dares deleting the variables we pass to it.
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mch_preclustering_bucket)

# Define application
O2_GENERATE_EXECUTABLE(
  EXE_NAME mch-preclusterizer-workflow

  SOURCES
  src/PreClusterizerWorkflow.cxx
  src/DigitSamplerSpec.cxx
  src/PreClusterFinderSpec.cxx
  src/PreClusterSinkSpec.cxx

  MODULE_LIBRARY_NAME ${LIBRARY_NAME}

  BUCKET_NAME ${BUCKET_NAME}
)
