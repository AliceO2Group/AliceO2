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
               SOURCES src/CalibRawBase.cxx
                       src/CalibPedestal.cxx
                       src/CalibPulser.cxx
                       src/CalibPedestalParam.cxx
                       src/CalibPulserParam.cxx
                       src/CalibTreeDump.cxx
                       src/DigitDump.cxx
               PUBLIC_LINK_LIBRARIES O2::DataFormatsTPC O2::TPCBase
                                     O2::TPCReconstruction ROOT::Minuit
                                     ms_gsl::ms_gsl)

o2_target_root_dictionary(TPCCalibration
                          HEADERS include/TPCCalibration/CalibRawBase.h
                                  include/TPCCalibration/CalibPedestal.h
                                  include/TPCCalibration/CalibPulser.h
                                  include/TPCCalibration/CalibPedestalParam.h
                                  include/TPCCalibration/CalibPulserParam.h
                                  include/TPCCalibration/CalibTreeDump.h
                                  include/TPCCalibration/DigitDump.h)

o2_add_executable(calib-pedestal
                  COMPONENT_NAME tpc
                  SOURCES run/calib-pedestal.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCCalibration O2::Framework O2::DPLUtils)

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
                       LABELS tpc)
o2_add_test_root_macro(macro/runPulser.C
                       PUBLIC_LINK_LIBRARIES O2::TPCCalibration
                       LABELS tpc)
o2_add_test_root_macro(macro/dumpDigits.C
                       PUBLIC_LINK_LIBRARIES O2::TPCCalibration
                       LABELS tpc)

