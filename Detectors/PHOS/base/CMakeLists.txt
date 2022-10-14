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

o2_add_library(PHOSBase
               SOURCES src/Geometry.cxx 
                       src/Hit.cxx 
                       src/PHOSSimParams.cxx
                       src/RCUTrailer.cxx
                       src/Mapping.cxx
               PUBLIC_LINK_LIBRARIES O2::MathUtils
                                     O2::SimulationDataFormat
                                     O2::SimConfig)

o2_target_root_dictionary(PHOSBase
                          HEADERS include/PHOSBase/Geometry.h
                                  include/PHOSBase/Hit.h
                                  include/PHOSBase/PHOSSimParams.h
                                  include/PHOSBase/RCUTrailer.h
                                  include/PHOSBase/Mapping.h)
o2_data_file(COPY files DESTINATION Detectors/PHOS)
