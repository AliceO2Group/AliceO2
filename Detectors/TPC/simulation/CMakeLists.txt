# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TPCSimulation
               SOURCES src/CommonMode.cxx
                       src/Detector.cxx
                       src/DigitMCMetaData.cxx
                       src/DigitContainer.cxx
                       src/DigitGlobalPad.cxx
                       src/Digitizer.cxx
                       src/DigitTime.cxx
                       src/ElectronTransport.cxx
                       src/GEMAmplification.cxx
                       src/PadResponse.cxx
                       src/Point.cxx
                       src/SAMPAProcessing.cxx
                       src/SpaceCharge.cxx
               PUBLIC_LINK_LIBRARIES O2::DetectorsBase O2::SimulationDataFormat
                                     O2::TPCBase O2::TPCSpaceChargeBase
                                     ROOT::Physics)

o2_target_root_dictionary(TPCSimulation
                          HEADERS include/TPCSimulation/CommonMode.h
                                  include/TPCSimulation/Detector.h
                                  include/TPCSimulation/DigitMCMetaData.h
                                  include/TPCSimulation/DigitContainer.h
                                  include/TPCSimulation/DigitGlobalPad.h
                                  include/TPCSimulation/Digitizer.h
                                  include/TPCSimulation/DigitTime.h
                                  include/TPCSimulation/ElectronTransport.h
                                  include/TPCSimulation/GEMAmplification.h
                                  include/TPCSimulation/PadResponse.h
                                  include/TPCSimulation/Point.h
                                  include/TPCSimulation/SAMPAProcessing.h
                                  include/TPCSimulation/SpaceCharge.h)

o2_data_file(COPY files DESTINATION Detectors/TPC)

if(BUILD_TESTING)
  add_subdirectory(test)

  o2_add_test_root_macro(macro/readMCtruth.C
                         PUBLIC_LINK_LIBRARIES O2::TPCBase
                                               O2::SimulationDataFormat
                         LABELS tpc)
endif()
