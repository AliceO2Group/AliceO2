set(MODULE_NAME "MFTWorkflow")
set(MODULE_BUCKET_NAME mft_workflow_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/RecoWorkflow.cxx
  src/DigitReaderSpec.cxx
  src/ClustererSpec.cxx
  src/ClusterWriterSpec.cxx
  src/ClusterReaderSpec.cxx
  src/TrackerSpec.cxx
  src/TrackWriterSpec.cxx
   )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

O2_GENERATE_EXECUTABLE(
  EXE_NAME "mft-reco-workflow"

  SOURCES
  src/mft-reco-workflow.cxx

  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

