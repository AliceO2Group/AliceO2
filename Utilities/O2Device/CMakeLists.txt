o2_add_library(O2Device
               SOURCES
               src/O2Device.cxx
               PUBLIC_LINK_LIBRARIES
               FairMQ::FairMQ
               O2::MemoryResources
               O2::Headers
               ms_gsl
               AliceO2::Monitoring)

o2_add_test(O2Device
            SOURCES
            test/test_O2Device.cxx
            PUBLIC_LINK_LIBRARIES
            O2::O2Device
            COMPONENT_NAME
            O2Device)
