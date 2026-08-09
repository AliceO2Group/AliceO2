# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(DataFormatsTRKFT3
               SOURCES src/Digit.cxx
                       src/Hit.cxx
                       src/ROFRecord.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonDataFormat
                                     O2::SimulationDataFormat)

o2_target_root_dictionary(DataFormatsTRKFT3
                          HEADERS include/DataFormatsTRKFT3/Cluster.h
                                  include/DataFormatsTRKFT3/Digit.h
                                  include/DataFormatsTRKFT3/Hit.h
                                  include/DataFormatsTRKFT3/ROFRecord.h
                          LINKDEF src/DataFormatsTRKFT3LinkDef.h)
