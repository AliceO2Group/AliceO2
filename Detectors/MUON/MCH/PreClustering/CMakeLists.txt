o2_add_library(MCHPreClustering SOURCES src/PreClusterFinder.cxx
                  PUBLIC_LINK_LIBRARIES
                  O2::MCHBase
                  O2::Framework)

o2_add_executable(preclusterizer-workflow
                  COMPONENT_NAME
                  mch
                  SOURCES
                  src/PreClusterizerWorkflow.cxx
                  src/DigitSamplerSpec.cxx
                  src/PreClusterFinderSpec.cxx
                  src/PreClusterSinkSpec.cxx
                  PUBLIC_LINK_LIBRARIES
                  O2::MCHPreClustering)
