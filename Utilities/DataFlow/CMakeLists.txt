# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# the bucket contains the following dependencies - common_boost_bucket - Base -
# Headers - O2device - dl the 'dl' dependency is needed as the device
# boilerplate code in runSimpleMQStateMachine.h uses dlopen etc. Probably this
# hidden dependency can be avoided by including the to some compiled FairMQ
# library

o2_add_library(DataFlow
               SOURCES src/FakeTimeframeBuilder.cxx
                       src/FakeTimeframeGeneratorDevice.cxx
                       src/HeartbeatSampler.cxx
                       src/SubframeBuilderDevice.cxx
                       src/TimeframeParser.cxx
                       src/TimeframeReaderDevice.cxx
                       src/TimeframeValidatorDevice.cxx
                       src/TimeframeWriterDevice.cxx
                       src/EPNReceiverDevice.cxx
                       src/FLPSenderDevice.cxx
               PUBLIC_LINK_LIBRARIES O2::Headers O2::TimeFrame FairMQ::FairMQ
                                     O2::Device)

o2_target_man_page(DataFlow NAME o2-timeframe-reader-device)
o2_target_man_page(DataFlow NAME o2-timeframe-writer-device)
o2_target_man_page(DataFlow NAME o2-subframebuilder-device)

o2_add_executable(fake-timeframegenerator-device
                  SOURCES src/runFakeTimeframeGeneratorDevice
                  PUBLIC_LINK_LIBRARIES O2::DataFlow)

o2_add_executable(heartbeat-sampler
                  SOURCES src/runHeartbeatSampler
                  PUBLIC_LINK_LIBRARIES O2::DataFlow)

o2_add_executable(subframebuilder-device
                  SOURCES src/runSubframeBuilderDevice
                  PUBLIC_LINK_LIBRARIES O2::DataFlow)

o2_add_executable(timeframe-reader-device
                  SOURCES src/runTimeframeReaderDevice
                  PUBLIC_LINK_LIBRARIES O2::DataFlow)

o2_add_executable(timeframe-validator-device
                  SOURCES src/runTimeframeValidatorDevice
                  PUBLIC_LINK_LIBRARIES O2::DataFlow)

o2_add_executable(timeframe-writer-device
                  SOURCES src/runTimeframeWriterDevice
                  PUBLIC_LINK_LIBRARIES O2::DataFlow)

o2_add_executable(epn-receiver-device
                  SOURCES src/runEPNReceiver
                  PUBLIC_LINK_LIBRARIES O2::DataFlow)

o2_add_executable(flp-sender-device
                  SOURCES src/runFLPSender
                  PUBLIC_LINK_LIBRARIES O2::DataFlow)

o2_add_executable(timeframe-validation-tool
                  SOURCES src/TimeframeValidationTool
                  PUBLIC_LINK_LIBRARIES O2::DataFlow)

o2_add_test(TimeframeParser
            SOURCES test/test_TimeframeParser
            PUBLIC_LINK_LIBRARIES O2::DataFlow
            COMPONENT_NAME dataflow
            LABELS utils)

o2_add_test(SubframeUtils01
            SOURCES test/test_SubframeUtils01
            PUBLIC_LINK_LIBRARIES O2::DataFlow
            COMPONENT_NAME dataflow
            LABELS utils)

o2_add_test(PayloadMerger01
            SOURCES test/test_PayloadMerger01
            PUBLIC_LINK_LIBRARIES O2::DataFlow
            COMPONENT_NAME dataflow
            LABELS utils)
