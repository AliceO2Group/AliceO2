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

o2_add_library(DataFormatsCPV
               SOURCES src/CPVBlockHeader.cxx
                       src/Hit.cxx
                       src/Digit.cxx
                       src/Cluster.cxx
                       src/TriggerRecord.cxx
                       src/CTF.cxx
                       src/CalibParams.cxx
                       src/BadChannelMap.cxx
                       src/Pedestals.cxx
                PUBLIC_LINK_LIBRARIES O2::CommonDataFormat
                                     O2::Headers
                                     O2::MathUtils
                                     O2::DetectorsBase
                                     O2::CPVBase
                                     O2::SimulationDataFormat
                                     Boost::serialization)

o2_target_root_dictionary(DataFormatsCPV
                          HEADERS include/DataFormatsCPV/CPVBlockHeader.h
                                  include/DataFormatsCPV/Hit.h
                                  include/DataFormatsCPV/Digit.h
                                  include/DataFormatsCPV/Cluster.h
                                  include/DataFormatsCPV/TriggerRecord.h
                                  include/DataFormatsCPV/CTF.h
                                  include/DataFormatsCPV/CalibParams.h
                                  include/DataFormatsCPV/BadChannelMap.h
                                  include/DataFormatsCPV/Pedestals.h)
