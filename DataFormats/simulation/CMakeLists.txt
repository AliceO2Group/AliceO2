# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(SimulationDataFormat
               SOURCES src/Stack.cxx
                       src/MCTrack.cxx
                       src/MCCompLabel.cxx
                       src/RunContext.cxx
                       src/StackParam.cxx
                       src/MCEventHeader.cxx
               PUBLIC_LINK_LIBRARIES ms_gsl::ms_gsl
                                     O2::DetectorsCommonDataFormats
                                     O2::GPUCommon O2::DetectorsBase
                                     O2::SimConfig)

o2_target_root_dictionary(
  SimulationDataFormat
  HEADERS include/SimulationDataFormat/Stack.h
          include/SimulationDataFormat/StackParam.h
          include/SimulationDataFormat/MCTrack.h
          include/SimulationDataFormat/BaseHits.h
          include/SimulationDataFormat/MCTruthContainer.h
          include/SimulationDataFormat/MCCompLabel.h
          include/SimulationDataFormat/TrackReference.h
          include/SimulationDataFormat/PrimaryChunk.h
          include/SimulationDataFormat/RunContext.h
          include/SimulationDataFormat/LabelContainer.h
          include/SimulationDataFormat/MCEventHeader.h
          include/SimulationDataFormat/MCEventStats.h
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

o2_add_test(MCTrack
            SOURCES test/MCTrack.cxx
            COMPONENT_NAME SimulationDataFormat
            PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat)
