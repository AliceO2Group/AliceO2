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

o2_add_library(GlobalTrackingWorkflowReaders
               SOURCES src/TrackCosmicsReaderSpec.cxx
                       src/SecondaryVertexReaderSpec.cxx
                       src/PrimaryVertexReaderSpec.cxx
                       src/StrangenessTrackingReaderSpec.cxx
                       src/TrackTPCITSReaderSpec.cxx
                       src/GlobalFwdTrackReaderSpec.cxx
                       src/MatchedMFTMCHReaderSpec.cxx
                       src/MatchedMCHMIDReaderSpec.cxx
                       src/IRFrameReaderSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::ReconstructionDataFormats
                                     O2::DataFormatsITSMFT
                                     O2::SimulationDataFormat)

o2_add_executable(reader-workflow
                  COMPONENT_NAME irframe
                  SOURCES src/irframe-reader-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflowReaders
                                        O2::DetectorsRaw)
