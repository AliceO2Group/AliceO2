o2_add_library(TOFBase
               SOURCES
               src/Geo.cxx
               src/Digit.cxx
               PUBLIC_LINK_LIBRARIES
               Boost::serialization
               FairRoot::Base
               O2::DetectorsBase)

o2_target_root_dictionary(TOFBase HEADERS include/TOFBase/Geo.h
                          include/TOFBase/Digit.h)

o2_add_test(TOFIndex SOURCES test/testTOFIndex.cxx COMPONENT_NAME TOF
        PUBLIC_LINK_LIBRARIES TOFBase)
