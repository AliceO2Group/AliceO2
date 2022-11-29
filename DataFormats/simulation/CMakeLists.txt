# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(SimulationDataFormat
               SOURCES src/MCTrack.cxx
                       src/MCCompLabel.cxx
                       src/MCEventLabel.cxx
                       src/DigitizationContext.cxx
                       src/StackParam.cxx
                       src/MCEventHeader.cxx
                       src/CustomStreamers.cxx
                       src/MCUtils.cxx
                       src/O2DatabasePDG.cxx
               PUBLIC_LINK_LIBRARIES Microsoft.GSL::GSL
                                     FairRoot::Base
                                     O2::DetectorsCommonDataFormats
                                     O2::DataFormatsParameters
                                     O2::GPUCommon
                                     ROOT::TreePlayer)

o2_target_root_dictionary(
  SimulationDataFormat
  HEADERS include/SimulationDataFormat/StackParam.h
          include/SimulationDataFormat/MCTrack.h
          include/SimulationDataFormat/BaseHits.h
          include/SimulationDataFormat/MCTruthContainer.h
          include/SimulationDataFormat/ConstMCTruthContainer.h
          include/SimulationDataFormat/MCCompLabel.h
          include/SimulationDataFormat/MCEventLabel.h
          include/SimulationDataFormat/TrackReference.h
          include/SimulationDataFormat/PrimaryChunk.h
          include/SimulationDataFormat/DigitizationContext.h
          include/SimulationDataFormat/LabelContainer.h
          include/SimulationDataFormat/MCEventHeader.h
          include/SimulationDataFormat/MCEventStats.h
          include/SimulationDataFormat/IOMCTruthContainerView.h
          include/SimulationDataFormat/MCUtils.h
          include/SimulationDataFormat/MCGenStatus.h
          include/SimulationDataFormat/O2DatabasePDG.h
  LINKDEF src/SimulationDataLinkDef.h)
# note the explicit LINKDEF as the linkdef in src is
#
# * src/SimulationDataLinkDef.h
# * and not src/SimulationDataFormatLinkDef.h

o2_add_test(BasicHits
            SOURCES test/testBasicHits.cxx
            COMPONENT_NAME SimulationDataFormat
            PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat)

o2_add_test(MCTruthContainer
            SOURCES test/testMCTruthContainer.cxx
            COMPONENT_NAME SimulationDataFormat
            PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat)

o2_add_test(MCCompLabel
            SOURCES test/testMCCompLabel.cxx
            COMPONENT_NAME SimulationDataFormat
            PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat)

o2_add_test(MCEventLabel
            SOURCES test/testMCEventLabel.cxx
            COMPONENT_NAME SimulationDataFormat
            PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat)

o2_add_test(MCTrack
            SOURCES test/MCTrack.cxx
            COMPONENT_NAME SimulationDataFormat
            PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat)
