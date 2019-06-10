o2_add_library(TimeFrame
               SOURCES src/TimeFrame.cxx
               PUBLIC_LINK_LIBRARIES FairMQ::FairMQ O2::Headers ROOT::RIO)

o2_target_root_dictionary(TimeFrame HEADERS include/TimeFrame/TimeFrame.h)

o2_add_test(TimeFrameTest
            SOURCES test/TimeFrameTest.cxx
            PUBLIC_LINK_LIBRARIES O2::TimeFrame
            COMPONENT_NAME DataFormats)
