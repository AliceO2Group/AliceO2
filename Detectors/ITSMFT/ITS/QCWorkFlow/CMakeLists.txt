set(MODULE_NAME "ITSQCWorkFlow")
set(MODULE_BUCKET_NAME ITS_qcworkflow_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/HisAnalyzerSpec.cxx
   #src/runBasic.cxx
   )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

O2_GENERATE_EXECUTABLE(
  EXE_NAME "its-qc-workflow"

  SOURCES
 # src/SimpleDigitizerWorkflow.cxx
  src/its-qc-workflow.cxx
 # src/SimReaderSpec.cxx
 # src/CollisionTimePrinter.cxx
 # src/TPCDigitizerSpec.cxx
 # src/TPCDigitRootWriterSpec.cxx
 # src/ITSMFTDigitizerSpec.cxx
 # src/ITSMFTDigitWriterSpec.cxx
 # src/TOFDigitizerSpec.cxx
 # src/TOFDigitWriterSpec.cxx
 # src/TOFClusterizerSpec.cxx
 # src/TOFClusterWriterSpec.cxx
 # src/FITDigitizerSpec.cxx
 # src/FITDigitWriterSpec.cxx
 # src/EMCALDigitizerSpec.cxx
 # src/EMCALDigitWriterSpec.cxx
 # src/HMPIDDigitizerSpec.cxx
 # src/MCHDigitizerSpec.cxx
 # src/TRDDigitizerSpec.cxx
 # src/GRPUpdaterSpec.cxx
  #src/its-qc-workflow.cxx
  #src/HisAnalyzerSpec.cxx
 
  
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

