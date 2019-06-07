# @author Matthias Richter @brief  cmake setup for module
# devices/aliceHLTwrapper

o2_add_library(aliceHLTwrapper
               SOURCES
               src/SystemInterface.cxx
               src/HOMERFactory.cxx
               src/WrapperDevice.cxx
               src/Component.cxx
               src/MessageFormat.cxx
               src/EventSampler.cxx
               PUBLIC_LINK_LIBRARIES
               FairMQ::FairMQ
               O2::Headers
               O2::O2Device
               Boost::thread)

o2_add_executable(wrapper-device
                  SOURCES
                  src/aliceHLTWrapper.cxx
                  COMPONENT_NAME
                  alicehlt
                  PUBLIC_LINK_LIBRARIES
                  O2::aliceHLTwrapper)

o2_add_executable(eventsampler-device
                  SOURCES
                  src/aliceHLTEventSampler.cxx
                  COMPONENT_NAME
                  alicehlt
                  PUBLIC_LINK_LIBRARIES
                  O2::aliceHLTwrapper)

o2_add_executable(runcomponent
                  SOURCES
                  src/runComponent.cxx
                  COMPONENT_NAME
                  alicehlt
                  PUBLIC_LINK_LIBRARIES
                  O2::aliceHLTwrapper)

o2_add_test(MessageFormat
            COMPONENT_NAME
            alicehlt
            SOURCES
            test/testMessageFormat.cxx
            PUBLIC_LINK_LIBRARIES
            O2::aliceHLTwrapper)

# FIXME: reconnect those
#
# * O2_GENERATE_MAN(NAME o2-alicehlt-wrapper-device)
# * O2_GENERATE_MAN(NAME AliceHLTComponents)
