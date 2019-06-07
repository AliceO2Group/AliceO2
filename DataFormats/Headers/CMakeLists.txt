o2_add_library(Headers
               SOURCES
               src/DataHeader.cxx
               src/NameHeader.cxx
               src/HeartbeatFrame.cxx
               src/TimeStamp.cxx
               PUBLIC_LINK_LIBRARIES
               O2::MemoryResources)

o2_add_test(DataHeader SOURCES test/testDataHeader.cxx PUBLIC_LINK_LIBRARIES
            O2::Headers)

o2_add_test(TimeStamp SOURCES test/testTimeStamp.cxx PUBLIC_LINK_LIBRARIES
            O2::Headers)

o2_add_test(HeartbeatFrame SOURCES test/test_HeartbeatFrame.cxx
            PUBLIC_LINK_LIBRARIES O2::Headers)

o2_add_test(RAWDataHeader SOURCES test/test_RAWDataHeader.cxx
            PUBLIC_LINK_LIBRARIES O2::Headers)
