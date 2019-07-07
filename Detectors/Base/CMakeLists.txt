# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DetectorsBase
               SOURCES src/Detector.cxx
                       src/GeometryManager.cxx
                       src/MaterialManager.cxx
                       src/Propagator.cxx
                       src/MatLayerCyl.cxx
                       src/MatLayerCylSet.cxx
                       src/Ray.cxx
               PUBLIC_LINK_LIBRARIES FairRoot::Base
                                     O2::CommonUtils
                                     O2::DetectorsCommonDataFormats
                                     O2::GPUCommon
                                     O2::ReconstructionDataFormats
                                     O2::Field
                                     FairMQ::FairMQ
                                     O2::DataFormatsParameters
                                     ROOT::VMC)

o2_target_root_dictionary(DetectorsBase
                          HEADERS include/DetectorsBase/Detector.h
                                  include/DetectorsBase/GeometryManager.h
                                  include/DetectorsBase/MaterialManager.h
                                  include/DetectorsBase/Propagator.h
                                  include/DetectorsBase/Triggers.h
                                  include/DetectorsBase/Ray.h
                                  include/DetectorsBase/MatCell.h
                                  include/DetectorsBase/MatLayerCyl.h
                                  include/DetectorsBase/MatLayerCylSet.h)

if(BUILD_SIMULATION)
  o2_add_test(
    MatBudLUT
    SOURCES test/testMatBudLUT.cxx
    COMPONENT_NAME DetectorsBase
    PUBLIC_LINK_LIBRARIES O2::DetectorsBase
    LABELS detectorsbase
    ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
                VMCWORKDIR=${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_DATADIR})
endif()

o2_add_test_root_macro(test/buildMatBudLUT.C
                       PUBLIC_LINK_LIBRARIES O2::DetectorsBase
                       LABELS detectorsbase)
