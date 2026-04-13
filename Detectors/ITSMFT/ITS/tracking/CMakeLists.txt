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

o2_add_library(ITStracking
               TARGETVARNAME targetName
               SOURCES src/ClusterLines.cxx
                       src/Cluster.cxx
                       src/Configuration.cxx
                       src/FastMultEstConfig.cxx
                       src/FastMultEst.cxx
                       src/TimeFrame.cxx
                       src/IOUtils.cxx
                       src/Tracker.cxx
                       src/TrackerTraits.cxx
                       src/TrackingConfigParam.cxx
                       src/Vertexer.cxx
                       src/VertexerTraits.cxx
               PUBLIC_LINK_LIBRARIES
                       O2::GPUCommon
                       Microsoft.GSL::GSL
                       O2::CommonConstants
                       O2::DataFormatsITSMFT
                       O2::SimulationDataFormat
                       O2::ITSBase
                       O2::CommonUtils
                       O2::ITSReconstruction
                       O2::ITSMFTReconstruction
                       O2::DataFormatsITS
               PRIVATE_LINK_LIBRARIES
                       O2::Steer
                       TBB::tbb)
# target_compile_options(${targetName} PRIVATE -O0 -g -fPIC -fno-omit-frame-pointer)

o2_add_library(ITSTrackingInterface
               TARGETVARNAME targetName
               SOURCES src/TrackingInterface.cxx
               PRIVATE_LINK_LIBRARIES
                       O2::ITStracking
                       O2::Framework
                       O2::GPUTracking)

o2_target_root_dictionary(ITStracking
                          HEADERS include/ITStracking/ClusterLines.h
                                  include/ITStracking/Tracklet.h
                                  include/ITStracking/Cluster.h
                                  include/ITStracking/Definitions.h
                                  include/ITStracking/FastMultEst.h
                                  include/ITStracking/FastMultEstConfig.h
                                  include/ITStracking/TrackingConfigParam.h
                          LINKDEF src/TrackingLinkDef.h)

if(CUDA_ENABLED OR HIP_ENABLED)
  add_subdirectory(GPU)
endif()

add_subdirectory(test)
