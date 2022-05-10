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

o2_add_library(FT0Base
               SOURCES src/Geometry.cxx
               PUBLIC_LINK_LIBRARIES ROOT::Physics FairRoot::Base O2::DetectorsBase O2::DetectorsCommonDataFormats O2::FrameworkLogger)

o2_target_root_dictionary(FT0Base HEADERS include/FT0Base/Geometry.h
                                          include/FT0Base/Constants.h
                                          include/FT0Base/FT0DigParam.h)

o2_data_file(COPY files DESTINATION Detectors/FT0/)
