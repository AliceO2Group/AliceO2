set(MODULE_NAME "RawPixelWorkFlow")
set(MODULE_BUCKET_NAME ITS_RawPixel_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/RawPixelWorkflow.cxx
  src/RawPixelReaderSpec.cxx
  src/RawPixelGetterSpec.cxx
 
   )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

O2_GENERATE_EXECUTABLE(
  EXE_NAME "its-raw-workflow"

  SOURCES
  src/its-raw-workflow.cxx

  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

