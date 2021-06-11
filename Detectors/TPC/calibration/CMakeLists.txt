# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

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
                       src/IDCAverageGroup.cxx
                       src/IDCGroup.cxx
                       src/IDCGroupHelperRegion.cxx
                       src/IDCGroupingParameter.cxx
                       src/IDCFactorization.cxx
                       src/IDCFourierTransform.cxx
                       src/RobustAverage.cxx
                       src/IDCCCDBHelper.cxx
               PUBLIC_LINK_LIBRARIES O2::DataFormatsTPC O2::TPCBase
                                     O2::TPCReconstruction ROOT::Minuit
                                     Microsoft.GSL::GSL)

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
                                  include/TPCCalibration/FastHisto.h
                                  include/TPCCalibration/IDCAverageGroup.h
                                  include/TPCCalibration/IDCGroup.h
                                  include/TPCCalibration/IDCGroupHelperRegion.h
                                  include/TPCCalibration/IDCGroupHelperSector.h
                                  include/TPCCalibration/IDCFactorization.h
                                  include/TPCCalibration/IDCGroupingParameter.h
                                  include/TPCCalibration/IDCContainer.h
                                  include/TPCCalibration/RobustAverage.h
                                  include/TPCCalibration/IDCFourierTransform.h
                                  include/TPCCalibration/IDCCCDBHelper.h)

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
o2_add_test_root_macro(macro/extractGainMap.C
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

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
