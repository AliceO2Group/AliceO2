set(MODULE_NAME "ITSDIGIRECOWorkflow")
set(MODULE_BUCKET_NAME ITS_digitreco_workflow_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/ITSMFTDigitizerSpec.cxx	
  src/DIGIRECOWorkFlow.cxx
  src/ClustererSpec.cxx
  src/TrackerSpec.cxx
  src/CookedTrackerSpec.cxx
  src/TrackWriterSpec.cxx
   )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

O2_GENERATE_EXECUTABLE(
  EXE_NAME "its-digi-reco-workflow"

  SOURCES
 # src/SimpleDigitizerWorkflow.cxx 
  src/its-digi-reco-workflow.cxx

  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

