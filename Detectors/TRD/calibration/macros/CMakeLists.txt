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

if(BUILD_TESTING)
o2_add_test_root_macro(checkNoisyMCMs.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsTRD
                                             O2::CCDB
                       LABELS trd)
endif()

o2_add_test_root_macro(makeTRDCCDBEntryForDCS.C
                       PUBLIC_LINK_LIBRARIES O2::DetectorsDCS
                                             O2::CCDB
                       LABELS trd)

o2_add_test_root_macro(readTRDDCSentries.C
                       PUBLIC_LINK_LIBRARIES O2::DetectorsDCS
                                             O2::CCDB
                       LABELS trd)

o2_add_test_root_macro(readTRDVoltages.C
                      PUBLIC_LINK_LIBRARIES O2::DetectorsDCS
                                            O2::DataFormatsTRD
                                            O2::CCDB
                      LABELS trd COMPILE_ONLY)

o2_add_test_root_macro(trdPadNoise.C
                      PUBLIC_LINK_LIBRARIES O2::DataFormatsTRD
                                            O2::TRDCalibration
                      LABELS trd COMPILE_ONLY)

o2_add_test_root_macro(manualCalibFit.C
                      PUBLIC_LINK_LIBRARIES O2::DataFormatsTRD
                                            O2::TRDCalibration
                      LABELS trd COMPILE_ONLY)

o2_add_test_root_macro(plotVdriftExB.C
                      PUBLIC_LINK_LIBRARIES O2::DataFormatsTRD
                                            O2::CCDB
                      LABELS trd)

o2_add_test_root_macro(makeDeflectionCorrelation.C
                      PUBLIC_LINK_LIBRARIES O2::DataFormatsTRD
                                            O2::CCDB
                                            O2::TRDQC
                      LABELS trd COMPILE_ONLY)

install(
  FILES makeTRDCCDBEntryForDCS.C
  readTRDDCSentries.C
  readTRDVoltages.C
  manualCalibFit.C
  plotVdriftExB.C
  makeDeflectionCorrelation.C
  DESTINATION share/macro/)
