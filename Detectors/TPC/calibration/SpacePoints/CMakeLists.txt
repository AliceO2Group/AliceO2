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
# add_compile_options(-O0 -g -fPIC)

o2_add_library(SpacePoints
               SOURCES src/SpacePointsCalibParam.cxx
                       src/TrackResiduals.cxx
                       src/TrackInterpolation.cxx
                       src/ResidualAggregator.cxx
                       src/SpacePointsCalibConfParam.cxx
               PUBLIC_LINK_LIBRARIES O2::DataFormatsTPC
                                     O2::CommonUtils
                                     O2::TPCBase
                                     O2::TRDBase
                                     O2::TOFBase
                                     O2::TPCReconstruction
                                     O2::TPCFastTransformation
                                     O2::ITStracking
                                     O2::DetectorsCalibration
                                     O2::DataFormatsITS
                                     O2::DataFormatsITSMFT
                                     O2::DataFormatsTRD
                                     O2::DataFormatsTOF
                                     O2::DataFormatsGlobalTracking
                                     O2::GPUTracking)

o2_target_root_dictionary(SpacePoints
                          HEADERS include/SpacePoints/TrackResiduals.h
                                  include/SpacePoints/TrackInterpolation.h
                                  include/SpacePoints/ResidualAggregator.h
                                  include/SpacePoints/SpacePointsCalibConfParam.h
                          LINKDEF src/SpacePointCalibLinkDef.h)

o2_add_test_root_macro(macro/staticMapCreator.C
                       PUBLIC_LINK_LIBRARIES O2::SpacePoints
                       LABELS tpc COMPILE_ONLY)

o2_add_test_root_macro(macro/staticMapCreatorCPM.C
                       PUBLIC_LINK_LIBRARIES O2::SpacePoints
                                             O2::CCDB
                                             O2::Algorithm
                                             O2::Framework
                                             O2::CommonConstants
                                             O2::DataFormatsParameters
                                             O2::ReconstructionDataFormats
                                             O2::DetectorsBase
                       LABELS tpc COMPILE_ONLY)

o2_add_test_root_macro(macro/SmoothingExtrapolate.C
                       PUBLIC_LINK_LIBRARIES O2::SpacePoints
                                             O2::Algorithm
                                             O2::TPCCalibration
                                             O2::CCDB
                                             O2::TPCBaseRecSim
                       LABELS tpc COMPILE_ONLY)

o2_add_test_root_macro(macro/voxResQA.C
                       PUBLIC_LINK_LIBRARIES O2::SpacePoints
                                             O2::TPCBaseRecSim
                                             O2::CommonUtils
                                             O2::Framework
                       LABELS tpc COMPILE_ONLY)

install(FILES macro/staticMapCreator.C
        DESTINATION share/macro/)

install(FILES macro/staticMapCreatorCPM.C
              macro/SmoothingExtrapolate.C
              macro/voxResQA.C
        DESTINATION share/macro/)

o2_add_test(TrackResiduals
COMPONENT_NAME calibration
PUBLIC_LINK_LIBRARIES O2::SpacePoints
SOURCES test/testTrackResiduals.cxx
ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
LABELS tpc
CONFIGURATIONS RelWithDebInfo Release MinSizeRel)
