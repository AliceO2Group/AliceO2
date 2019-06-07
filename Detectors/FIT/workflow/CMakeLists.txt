o2_add_library(FITWorkflow
               SOURCES
               src/RecoWorkflow.cxx
               src/T0DigitReaderSpec.cxx
               src/T0ReconstructorSpec.cxx
               src/T0RecPointWriterSpec.cxx
               src/T0RecPointReaderSpec.cxx
               PUBLIC_LINK_LIBRARIES 
               O2::T0Reconstruction
               O2::Framework)

o2_add_executable(reco-workflow
                  COMPONENT_NAME
                  fit
                  SOURCES
                  src/fit-reco-workflow.cxx
                  PUBLIC_LINK_LIBRARIES
                  O2::FITWorkflow)
