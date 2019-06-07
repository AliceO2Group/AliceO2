o2_add_library(ITSMFTWorkflow SOURCES src/ClusterReaderSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
               O2::DataFormatsITSMFT
               O2::SimulationDataFormat)
