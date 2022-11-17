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

o2_add_library(ITSSimulation
               SOURCES src/V11Geometry.cxx src/V1Layer.cxx src/V3Layer.cxx
                       src/Detector.cxx src/V3Services.cxx src/V3Cage.cxx
                       src/DescriptorInnerBarrelITS2.cxx
               PUBLIC_LINK_LIBRARIES O2::ITSBase O2::ITSMFTSimulation ROOT::Physics
                                     $<$<BOOL:${ENABLE_UPGRADES}>:O2::ITS3Base> $<$<BOOL:${ENABLE_UPGRADES}>:O2::ITS3Simulation>)

o2_target_root_dictionary(ITSSimulation
                          HEADERS include/ITSSimulation/Detector.h
                                  include/ITSSimulation/V1Layer.h
                                  include/ITSSimulation/V3Layer.h
                                  include/ITSSimulation/V3Cage.h
                                  include/ITSSimulation/V11Geometry.h
                                  include/ITSSimulation/V3Services.h
                                  include/ITSSimulation/DescriptorInnerBarrelITS2.h)

o2_data_file(COPY data  DESTINATION Detectors/ITS/simulation)

o2_add_executable(digi2raw
                  COMPONENT_NAME its
                  TARGETVARNAME itsdigi2raw_exe
                  SOURCES src/digi2raw.cxx
                  PUBLIC_LINK_LIBRARIES O2::ITSMFTReconstruction
                                        O2::DataFormatsITSMFT
                                        O2::ITSMFTBase
                                        O2::ITSMFTSimulation
                                        O2::DetectorsRaw
                                        O2::DetectorsCommonDataFormats
                                        O2::CommonUtils
                                        Boost::program_options)

if(NOT APPLE)

 set_property(TARGET ${itsdigi2raw_exe} PROPERTY LINK_WHAT_YOU_USE ON)

endif()
