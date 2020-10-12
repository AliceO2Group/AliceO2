# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DetectorsVertexing
               SOURCES src/DCAFitterN.cxx
                       src/PVertexer.cxx
                       src/PVertexerHelpers.cxx
                       src/PVertexerParams.cxx
                       src/VertexTrackMatcher.cxx
                       src/SVertexer.cxx
                       src/SVertexerParams.cxx
                       src/V0Hypothesis.cxx
               PUBLIC_LINK_LIBRARIES ROOT::Core
	                             O2::CommonUtils
                                     O2::ReconstructionDataFormats
                                     O2::DataFormatsTPC
                                     O2::DataFormatsITS
                                     O2::TPCBase
                                     O2::SimulationDataFormat
                                     O2::FT0Reconstruction
                                     O2::DataFormatsFT0
                                     ms_gsl::ms_gsl)

o2_target_root_dictionary(DetectorsVertexing
                          HEADERS include/DetectorsVertexing/HelixHelper.h
                                  include/DetectorsVertexing/PVertexer.h
                                  include/DetectorsVertexing/PVertexerHelpers.h
                                  include/DetectorsVertexing/PVertexerParams.h
                                  include/DetectorsVertexing/VertexTrackMatcher.h
                                  include/DetectorsVertexing/DCAFitterN.h
                                  include/DetectorsVertexing/SVertexer.h
                                  include/DetectorsVertexing/SVertexerParams.h
                                  include/DetectorsVertexing/V0Hypothesis.h)

o2_add_test(
  DCAFitterN
  SOURCES test/testDCAFitterN.cxx
  COMPONENT_NAME DetectorsVertexing
  PUBLIC_LINK_LIBRARIES O2::DetectorsVertexing ROOT::Core ROOT::Physics
  LABELS vertexing
  ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
  VMCWORKDIR=${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_DATADIR})
