o2_add_library(MessageMonitor
               SOURCES
               src/O2MessageMonitor.cxx
               PUBLIC_LINK_LIBRARIES
               FairLogger::FairLogger
               FairMQ::FairMQ
               O2::Device)

o2_add_executable(message-monitor SOURCES src/runO2MessageMonitor.cxx
        PUBLIC_LINK_LIBRARIES O2::MessageMonitor)

o2_add_test(O2MessageMonitorTest SOURCES test/O2MessageMonitorTest.cxx
            PUBLIC_LINK_LIBRARIES O2::MessageMonitor)
