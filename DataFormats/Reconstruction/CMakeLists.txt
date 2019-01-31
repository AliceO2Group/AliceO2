set(MODULE_NAME "ReconstructionDataFormats")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Track.cxx
  src/BaseCluster.cxx
  src/TrackTPCITS.cxx
  src/Vertex.cxx
  src/MatchInfoTOF.cxx
  src/TrackLTIntegral.cxx
  src/PID.cxx
 )

Set(HEADERS
  include/${MODULE_NAME}/Track.h
  include/${MODULE_NAME}/BaseCluster.h
  include/${MODULE_NAME}/TrackTPCITS.h
  include/${MODULE_NAME}/Vertex.h
  include/${MODULE_NAME}/MatchInfoTOF.h
  include/${MODULE_NAME}/TrackLTIntegral.h
  include/${MODULE_NAME}/PID.h
)

Set(LINKDEF src/ReconstructionDataFormatsLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME data_format_reconstruction_bucket)

O2_GENERATE_LIBRARY()


set(TEST_SRCS
  test/testVertex.cxx
  test/testLTOFIntegration.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
