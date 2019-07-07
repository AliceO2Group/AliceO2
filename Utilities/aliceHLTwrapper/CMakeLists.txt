# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# @author Matthias Richter @brief  cmake setup for module
# devices/aliceHLTwrapper

o2_add_library(aliceHLTwrapper
               SOURCES src/SystemInterface.cxx
                       src/HOMERFactory.cxx
                       src/WrapperDevice.cxx
                       src/Component.cxx
                       src/MessageFormat.cxx
                       src/EventSampler.cxx
               PUBLIC_LINK_LIBRARIES FairMQ::FairMQ O2::Headers O2::Device
                                     Boost::thread)

o2_target_man_page(aliceHLTwrapper NAME o2-alicehlt-wrapper-device)
o2_target_man_page(aliceHLTwrapper NAME AliceHLTComponents)

o2_add_executable(wrapper-device
                  SOURCES src/aliceHLTWrapper.cxx
                  COMPONENT_NAME alicehlt
                  PUBLIC_LINK_LIBRARIES O2::aliceHLTwrapper)

o2_add_executable(eventsampler-device
                  SOURCES src/aliceHLTEventSampler.cxx
                  COMPONENT_NAME alicehlt
                  PUBLIC_LINK_LIBRARIES O2::aliceHLTwrapper)

o2_add_executable(runcomponent
                  SOURCES src/runComponent.cxx
                  COMPONENT_NAME alicehlt
                  PUBLIC_LINK_LIBRARIES O2::aliceHLTwrapper)

o2_add_test(MessageFormat
            COMPONENT_NAME alicehlt
            SOURCES test/testMessageFormat.cxx
            PUBLIC_LINK_LIBRARIES O2::aliceHLTwrapper
            LABELS utils)
