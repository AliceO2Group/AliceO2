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

o2_add_library(TRDPID
               SOURCES src/PIDBase.cxx
                       src/ML.cxx
                       src/PIDParameters.cxx
               PRIVATE_LINK_LIBRARIES O2::TRDBase
                                     O2::DataFormatsTRD
                                     O2::DataFormatsGlobalTracking
                                     O2::DetectorsBase
                                     O2::MathUtils
                                     O2::GPUTracking
                                     O2::Framework
                                     O2::ReconstructionDataFormats
                                     fmt::fmt
                                     ONNXRuntime::ONNXRuntime)

o2_target_root_dictionary(TRDPID
                          HEADERS include/TRDPID/PIDBase.h
                                  include/TRDPID/PIDParameters.h
                                  include/TRDPID/ML.h
                                  include/TRDPID/Dummy.h)

add_subdirectory(macros)
