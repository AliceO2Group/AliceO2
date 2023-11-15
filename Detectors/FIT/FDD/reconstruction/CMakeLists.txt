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

o2_add_library(FDDReconstruction
               SOURCES src/Reconstructor.cxx
                       src/CTFCoder.cxx
               PUBLIC_LINK_LIBRARIES O2::FDDBase
                                     O2::DataFormatsFDD
                                     O2::DetectorsBase
                                     O2::DetectorsRaw)

o2_target_root_dictionary(FDDReconstruction
  HEADERS include/FDDReconstruction/Reconstructor.h)

o2_add_executable(
  test-raw-conversion
  COMPONENT_NAME fdd
  SOURCES src/test-raw-conversion.cxx
  PUBLIC_LINK_LIBRARIES O2::FDDReconstruction)

o2_add_executable(
  test-raw2digit
  COMPONENT_NAME fdd
  SOURCES src/test-raw2digit.cxx
  PUBLIC_LINK_LIBRARIES O2::FDDReconstruction)
