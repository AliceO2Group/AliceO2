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

o2_add_library(DataFormatsECal
               SOURCES src/Digit.cxx
               SOURCES src/MCLabel.cxx
               SOURCES src/Cluster.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonDataFormat
                                     O2::SimulationDataFormat
                                     AliceO2::InfoLogger)

o2_target_root_dictionary(DataFormatsECal
                          HEADERS include/DataFormatsECal/Digit.h
                          HEADERS include/DataFormatsECal/MCLabel.h
                          HEADERS include/DataFormatsECal/Cluster.h)
