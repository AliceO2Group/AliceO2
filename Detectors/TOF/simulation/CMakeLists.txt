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

o2_add_library(TOFSimulation
               SOURCES src/Detector.cxx src/Digitizer.cxx src/DigitizerTask.cxx src/TOFSimParams.cxx
               PUBLIC_LINK_LIBRARIES O2::DetectorsBase O2::TOFBase
                                     O2::SimulationDataFormat
				     O2::TOFCalibration)

o2_target_root_dictionary(TOFSimulation
                          HEADERS include/TOFSimulation/Detector.h
                                  include/TOFSimulation/Digitizer.h
                                  include/TOFSimulation/DigitizerTask.h
                                  include/TOFSimulation/TOFSimParams.h)

o2_add_executable(digi2raw
                  COMPONENT_NAME tof
                  SOURCES src/digi2raw.cxx
                  PUBLIC_LINK_LIBRARIES O2::CommonUtils
                                        Boost::program_options)
