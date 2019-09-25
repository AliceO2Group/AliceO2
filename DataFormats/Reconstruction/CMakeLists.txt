# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(ReconstructionDataFormats
               SOURCES src/Track.cxx
                       src/BaseCluster.cxx
                       src/TrackTPCITS.cxx
                       src/Vertex.cxx
                       src/MatchInfoTOF.cxx
                       src/TrackLTIntegral.cxx
                       src/PID.cxx
               PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                     O2::DetectorsCommonDataFormats
                                     O2::CommonDataFormat)

o2_target_root_dictionary(
  ReconstructionDataFormats
  HEADERS include/ReconstructionDataFormats/Track.h
          include/ReconstructionDataFormats/BaseCluster.h
          include/ReconstructionDataFormats/TrackTPCITS.h
          include/ReconstructionDataFormats/Vertex.h
          include/ReconstructionDataFormats/MatchInfoTOF.h
          include/ReconstructionDataFormats/TrackLTIntegral.h
          include/ReconstructionDataFormats/PID.h)

o2_add_test(Vertex
            SOURCES test/testVertex.cxx
            COMPONENT_NAME ReconstructionDataFormats
            PUBLIC_LINK_LIBRARIES O2::ReconstructionDataFormats
                                  O2::CommonDataFormat)

o2_add_test(LTOFIntegration
            SOURCES test/testLTOFIntegration.cxx
            COMPONENT_NAME ReconstructionDataFormats
            PUBLIC_LINK_LIBRARIES O2::ReconstructionDataFormats)
