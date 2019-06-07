o2_add_library(O2MessageMonitor
               SOURCES
               src/O2MessageMonitor.cxx
               PUBLIC_LINK_LIBRARIES
               FairLogger::FairLogger
               FairMQ::FairMQ
               O2::O2Device)

o2_add_executable(message-monitor SOURCES src/runO2MessageMonitor.cxx
        PUBLIC_LINK_LIBRARIES O2::O2MessageMonitor)

o2_add_test(O2MessageMonitorTest SOURCES test/O2MessageMonitorTest.cxx
            PUBLIC_LINK_LIBRARIES O2::O2MessageMonitor)
