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

o2_add_library(TPCSimulation
               SOURCES src/CommonMode.cxx
                       src/Detector.cxx
                       src/DigitContainer.cxx
                       src/DigitGlobalPad.cxx
                       src/Digitizer.cxx
                       src/DigitTime.cxx
                       src/ElectronTransport.cxx
                       src/GEMAmplification.cxx
                       src/Point.cxx
                       src/SAMPAProcessing.cxx
                       src/IDCSim.cxx
               PUBLIC_LINK_LIBRARIES O2::DetectorsBase O2::SimulationDataFormat
                                     O2::TPCBase O2::TPCSpaceCharge O2::TPCCalibration
                                     ROOT::Physics)

o2_target_root_dictionary(TPCSimulation
                          HEADERS include/TPCSimulation/CommonMode.h
                                  include/TPCSimulation/Detector.h
                                  include/TPCSimulation/DigitContainer.h
                                  include/TPCSimulation/DigitGlobalPad.h
                                  include/TPCSimulation/Digitizer.h
                                  include/TPCSimulation/DigitTime.h
                                  include/TPCSimulation/ElectronTransport.h
                                  include/TPCSimulation/GEMAmplification.h
                                  include/TPCSimulation/Point.h
                                  include/TPCSimulation/SAMPAProcessing.h
                                  include/TPCSimulation/IDCSim.h)

o2_data_file(COPY files DESTINATION Detectors/TPC)
o2_data_file(COPY data  DESTINATION Detectors/TPC/simulation)

if(BUILD_TESTING)
  add_subdirectory(test)

  o2_add_test_root_macro(macro/readMCtruth.C
                         PUBLIC_LINK_LIBRARIES O2::TPCBase
                                               O2::SimulationDataFormat
                         LABELS tpc)

  o2_add_test_root_macro(macro/laserTrackGenerator.C
                         PUBLIC_LINK_LIBRARIES FairRoot::Base
                                               O2::DataFormatsTPC
                         LABELS tpc)

  o2_add_test_root_macro(macro/toyCluster.C
                         PUBLIC_LINK_LIBRARIES O2::TPCBase
                                               O2::DataFormatsTPC
                         LABELS tpc)

endif()
