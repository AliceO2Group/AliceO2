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

o2_add_library(DetectorsBase
               SOURCES src/Detector.cxx
                       src/GeometryManager.cxx
                       src/MaterialManager.cxx
                       src/MaterialManagerParam.cxx
                       src/Propagator.cxx
                       src/MatLayerCyl.cxx
                       src/MatLayerCylSet.cxx
                       src/Ray.cxx
                       src/BaseDPLDigitizer.cxx
                       src/CTFCoderBase.cxx
                       src/Aligner.cxx
                       src/SimFieldUtils.cxx
                       src/TFIDInfoHelper.cxx
                       src/GRPGeomHelper.cxx
                       src/Stack.cxx
                       src/VMCSeederService.cxx
               PUBLIC_LINK_LIBRARIES FairRoot::Base
                                     O2::CommonUtils
                                     O2::DetectorsCommonDataFormats
                                     O2::DataFormatsCTP
                                     O2::GPUCommon
                                     O2::GPUUtils
                                     O2::ReconstructionDataFormats
                                     O2::Field
                                     O2::Framework
                                     FairMQ::FairMQ
                                     O2::DataFormatsParameters
                                     O2::SimulationDataFormat
                                     O2::SimConfig
                                     O2::CCDB
                                     MC::VMC
                                     TBB::tbb
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Merger # Must not link to avoid cyclic dependency
                             )

o2_target_root_dictionary(DetectorsBase
                          HEADERS include/DetectorsBase/Detector.h
                                  include/DetectorsBase/GeometryManager.h
                                  include/DetectorsBase/MaterialManager.h
                                  include/DetectorsBase/MaterialManagerParam.h
                                  include/DetectorsBase/Propagator.h
                                  include/DetectorsBase/Ray.h
                                  include/DetectorsBase/MatCell.h
                                  include/DetectorsBase/MatLayerCyl.h
                                  include/DetectorsBase/MatLayerCylSet.h
                                  include/DetectorsBase/Aligner.h
                                  include/DetectorsBase/Stack.h
                                  include/DetectorsBase/SimFieldUtils.h)

if(BUILD_SIMULATION)
  if (NOT APPLE)
    o2_add_test(
      MatBudLUT
      SOURCES test/testMatBudLUT.cxx
      COMPONENT_NAME DetectorsBase
      PUBLIC_LINK_LIBRARIES O2::DetectorsBase O2::ITSMFTReconstruction
      LABELS detectorsbase
      ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
                VMCWORKDIR=${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_DATADIR})
  endif()

  o2_add_test(
    MCStack
    SOURCES test/testStack.cxx
    COMPONENT_NAME DetectorsBase
    PUBLIC_LINK_LIBRARIES O2::DetectorsBase
    LABELS detectorsbase
    ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
                VMCWORKDIR=${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_DATADIR})
endif()

install(FILES test/buildMatBudLUT.C
              test/extractLUTLayers.C
              DESTINATION share/macro/)

o2_add_test_root_macro(test/buildMatBudLUT.C
                       PUBLIC_LINK_LIBRARIES O2::DetectorsBase
                       LABELS detectorsbase)

o2_add_test_root_macro(test/extractLUTLayers.C
                       PUBLIC_LINK_LIBRARIES O2::DetectorsBase
                       LABELS detectorsbase)
