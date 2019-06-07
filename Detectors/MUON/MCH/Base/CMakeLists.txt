o2_add_library(MCHBase
               SOURCES src/Mapping.cxx
                       src/DigitBlock.cxx
                       src/PreClusterBlock.cxx
                       src/ClusterBlock.cxx
                       src/TrackBlock.cxx
                       PUBLIC_LINK_LIBRARIES
                       ROOT::Core
                       FairRoot::Base
                       FairMQ::FairMQ)

o2_target_root_dictionary(MCHBase
                          HEADERS
                          include/MCHBase/Mapping.h
                          include/MCHBase/DigitBlock.h
                          include/MCHBase/PreClusterBlock.h
                          include/MCHBase/ClusterBlock.h
                          include/MCHBase/TrackBlock.h)
