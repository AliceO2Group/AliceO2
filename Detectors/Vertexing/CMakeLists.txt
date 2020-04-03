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
               PUBLIC_LINK_LIBRARIES ROOT::Core
	                             O2::CommonUtils
                                     O2::ReconstructionDataFormats)

o2_target_root_dictionary(DetectorsVertexing
                          HEADERS include/DetectorsVertexing/HelixHelper.h
                                  include/DetectorsVertexing/DCAFitterN.h)

o2_add_test(
  DCAFitterN
  SOURCES test/testDCAFitterN.cxx
  COMPONENT_NAME DetectorsVertexing
  PUBLIC_LINK_LIBRARIES O2::DetectorsVertexing ROOT::Core ROOT::Physics
  LABELS vertexing
  ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
  VMCWORKDIR=${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_DATADIR})
