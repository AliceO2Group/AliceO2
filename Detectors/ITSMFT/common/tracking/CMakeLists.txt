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

o2_add_library(ITSMFTTracking
               SOURCES src/BoundedAllocator.cxx
                       src/CapacityEstimator.cxx
                       src/ITSTrackingConfigParam.cxx
                       src/SlabBumpAllocator.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonConstants
                                     O2::CommonDataFormat
                                     O2::CommonUtils
                                     O2::DataFormatsITS
                                     O2::FrameworkLogger
                                     O2::GPUCommon
                                     O2::MathUtils
               PRIVATE_LINK_LIBRARIES
                                      TBB::tbb)

o2_target_root_dictionary(ITSMFTTracking
                          HEADERS include/ITSMFTTracking/ITSTrackingConfigParam.h
                          LINKDEF src/ITSTrackingLinkDef.h)

add_subdirectory(test)
