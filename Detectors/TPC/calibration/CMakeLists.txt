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

add_subdirectory(SpacePoints)

o2_add_library(TPCCalibration
               TARGETVARNAME targetName
               SOURCES src/CalibRawBase.cxx
                       src/CalibPedestal.cxx
                       src/CalibPulser.cxx
                       src/CalibPedestalParam.cxx
                       src/CalibPulserParam.cxx
                       src/CalibTreeDump.cxx
                       src/DigitDump.cxx
                       src/DigitDumpParam.cxx
                       src/CalibPadGainTracks.cxx
                       src/CalibPadGainTracksBase.cxx
                       src/CalibLaserTracks.cxx
                       src/LaserTracksCalibrator.cxx
                       src/SACDecoder.cxx
                       src/IDCAverageGroup.cxx
                       src/IDCAverageGroupBase.cxx
                       src/IDCAverageGroupHelper.cxx
                       src/IDCGroup.cxx
                       src/IDCGroupHelperRegion.cxx
                       src/IDCGroupHelperSector.cxx
                       src/IDCGroupingParameter.cxx
                       src/IDCFactorization.cxx
                       src/IDCFourierTransformBase.cxx
                       src/IDCFourierTransform.cxx
                       src/IDCDrawHelper.cxx
                       src/RobustAverage.cxx
                       src/IDCCCDBHelper.cxx
                       src/CalibdEdx.cxx
                       src/CalibratordEdx.cxx
                       src/TrackDump.cxx
                       src/CalibratorPadGainTracks.cxx
                       src/TPCVDriftTglCalibration.cxx
                       src/SACFactorization.cxx
                       src/SACParameter.cxx
                       src/SACDrawHelper.cxx
                       src/VDriftHelper.cxx
               PUBLIC_LINK_LIBRARIES O2::DataFormatsTPC O2::TPCBase
                                     O2::TPCReconstruction ROOT::Minuit
                                     Microsoft.GSL::GSL
                                     O2::DetectorsCalibration
                                     O2::GPUO2Interface
                                     O2::GPUTracking
                                     O2::CCDB)

o2_target_root_dictionary(TPCCalibration
                          HEADERS include/TPCCalibration/CalibRawBase.h
                                  include/TPCCalibration/CalibPedestal.h
                                  include/TPCCalibration/CalibPulser.h
                                  include/TPCCalibration/CalibPedestalParam.h
                                  include/TPCCalibration/CalibPulserParam.h
                                  include/TPCCalibration/CalibTreeDump.h
                                  include/TPCCalibration/DigitDump.h
                                  include/TPCCalibration/DigitDumpParam.h
                                  include/TPCCalibration/CalibPadGainTracks.h
                                  include/TPCCalibration/CalibPadGainTracksBase.h
                                  include/TPCCalibration/FastHisto.h
                                  include/TPCCalibration/CalibLaserTracks.h
                                  include/TPCCalibration/LaserTracksCalibrator.h
                                  include/TPCCalibration/SACDecoder.h
                                  include/TPCCalibration/IDCAverageGroup.h
                                  include/TPCCalibration/IDCAverageGroupBase.h
                                  include/TPCCalibration/IDCAverageGroupHelper.h
                                  include/TPCCalibration/IDCGroup.h
                                  include/TPCCalibration/IDCGroupHelperRegion.h
                                  include/TPCCalibration/IDCGroupHelperSector.h
                                  include/TPCCalibration/IDCFactorization.h
                                  include/TPCCalibration/IDCGroupingParameter.h
                                  include/TPCCalibration/IDCContainer.h
                                  include/TPCCalibration/RobustAverage.h
                                  include/TPCCalibration/IDCFourierTransformBase.h
                                  include/TPCCalibration/IDCFourierTransform.h
                                  include/TPCCalibration/IDCCCDBHelper.h
                                  include/TPCCalibration/CalibdEdx.h
                                  include/TPCCalibration/CalibratordEdx.h
                                  include/TPCCalibration/TrackDump.h
                                  include/TPCCalibration/IDCDrawHelper.h
                                  include/TPCCalibration/CalibratorPadGainTracks.h
                                  include/TPCCalibration/TPCVDriftTglCalibration.h
                                  include/TPCCalibration/SACFactorization.h
                                  include/TPCCalibration/SACParameter.h
                                  include/TPCCalibration/SACDrawHelper.h
                                  include/TPCCalibration/VDriftHelper.h)

o2_add_test_root_macro(macro/comparePedestalsAndNoise.C
                       PUBLIC_LINK_LIBRARIES O2::TPCBase
                       LABELS tpc)
o2_add_test_root_macro(macro/drawNoiseAndPedestal.C
                       PUBLIC_LINK_LIBRARIES O2::TPCBase
                       LABELS tpc)
o2_add_test_root_macro(macro/drawPulser.C
                       PUBLIC_LINK_LIBRARIES O2::TPCBase
                       LABELS tpc)
o2_add_test_root_macro(macro/mergeNoiseAndPedestal.C
                       PUBLIC_LINK_LIBRARIES O2::TPCBase
                       LABELS tpc)
o2_add_test_root_macro(macro/runPedestal.C
                       PUBLIC_LINK_LIBRARIES O2::TPCCalibration
                       LABELS tpc
                       ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)
o2_add_test_root_macro(macro/runPulser.C
                       PUBLIC_LINK_LIBRARIES O2::TPCCalibration
                       LABELS tpc
                       ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)
o2_add_test_root_macro(macro/dumpDigits.C
                       PUBLIC_LINK_LIBRARIES O2::TPCCalibration
                       LABELS tpc
                       ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)
o2_add_test_root_macro(macro/preparePedestalFiles.C
                       PUBLIC_LINK_LIBRARIES O2::TPCCalibration
                       LABELS tpc)

o2_add_test(IDCFourierTransform
            COMPONENT_NAME calibration
            PUBLIC_LINK_LIBRARIES O2::TPCCalibration
            SOURCES test/testO2TPCIDCFourierTransform.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            LABELS tpc
            CONFIGURATIONS RelWithDebInfo Release MinRelSize)

o2_add_test(IDCAverageGroup
            LABELS tpc
            PUBLIC_LINK_LIBRARIES O2::TPCCalibration
            COMPONENT_NAME tpc
            SOURCES test/testO2TPCIDCAverageGroup.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            CONFIGURATIONS RelWithDebInfo Release MinRelSize)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${targetName} PRIVATE FFTW3::fftw3f)
