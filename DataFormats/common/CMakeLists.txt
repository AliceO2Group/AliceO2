o2_add_library(CommonDataFormat
               SOURCES
               src/InteractionRecord.cxx
               src/BunchFilling.cxx
               PUBLIC_LINK_LIBRARIES
               O2::CommonConstants
               O2::GPUCommon
               ROOT::Core
               FairRoot::Base)

o2_target_root_dictionary(CommonDataFormat
                          HEADERS
                          include/CommonDataFormat/TimeStamp.h
                          include/CommonDataFormat/EvIndex.h
                          include/CommonDataFormat/RangeReference.h
                          include/CommonDataFormat/InteractionRecord.h
                          include/CommonDataFormat/BunchFilling.h)

o2_add_test(TimeStamp
            SOURCES
            test/testTimeStamp.cxx
            COMPONENT_NAME
            CommonDataFormat
            PUBLIC_LINK_LIBRARIES
            O2::CommonDataFormat)
