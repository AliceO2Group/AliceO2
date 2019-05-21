set(MODULE_NAME "TOFWorkflow")
set(MODULE_BUCKET_NAME TOF_workflow_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/RecoWorkflowSpec.cxx
  )

set(HEADERS
#  include/ReconstructionDataFormats/MatchInfoTOF.h
  include/${MODULE_NAME}/RecoWorkflowSpec.h
)

set(LINKDEF src/TOFWorkflowLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()
