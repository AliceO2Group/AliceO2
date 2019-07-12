o2_add_library(TestDataReaderWorkflow
               SOURCES src/TestDataReaderWorkflow.cxx
                       src/TestDataReader.cxx
                       src/TestDataGetter.cxx
                       PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::SimConfig
                                     O2::DataFormatsITS
                                     O2::SimulationDataFormat
                                     O2::ITStracking
                                     O2::ITSReconstruction)

o2_add_executable(qc-workflow
                  SOURCES src/its-qc-data-reader.cxx
                  COMPONENT_NAME its
                  PUBLIC_LINK_LIBRARIES O2::TestDataReaderWorkflow)
