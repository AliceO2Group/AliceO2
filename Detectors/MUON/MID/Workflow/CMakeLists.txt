set(MODULE_NAME "MIDWorkflow")
set(MODULE_BUCKET_NAME mid_workflow_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
        src/ClusterizerSpec.cxx
        src/ClusterLabelerSpec.cxx
        src/DigitReaderSpec.cxx
        src/RecoWorkflow.cxx
        src/TrackerSpec.cxx
        src/TrackLabelerSpec.cxx
        )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

O2_GENERATE_EXECUTABLE(
  EXE_NAME o2-mid-reco-workflow
  SOURCES src/mid-reco-workflow.cxx
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)