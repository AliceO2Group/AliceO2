# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(EC0tracking
               TARGETVARNAME targetName
               SOURCES src/ClusterLines.cxx
                       src/Cluster.cxx
                       src/ROframe.cxx
                       src/Graph.cxx
                       src/DBScan.cxx
                       src/IOUtils.cxx
                       src/Label.cxx
                       src/PrimaryVertexContext.cxx
                       src/Road.cxx
                       src/StandaloneDebugger.cxx
                       src/Tracker.cxx
                       src/TrackerTraitsEC0CPU.cxx
                       src/TrackingConfigParam.cxx
                       src/ClusterLines.cxx
                       src/Vertexer.cxx
                       src/VertexerTraitsEC0.cxx
               PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                     ms_gsl::ms_gsl
                                     O2::CommonConstants
                                     O2::DataFormatsITSMFT
                                     O2::DataFormatsEndCaps
                                     O2::SimulationDataFormat
                                     O2::ECLayersBase
                                     O2::ITStrackingCUDA
                                     O2::DataFormatsITS)

o2_target_root_dictionary(EC0tracking
                          HEADERS include/EC0tracking/ClusterLines.h
                                  include/EC0tracking/Tracklet.h
                                  include/EC0tracking/DBScan.h
                                  include/EC0tracking/TrackingConfigParam.h
                          LINKDEF src/TrackingLinkDef.h)

if(CUDA_ENABLED)
  add_subdirectory(cuda)
  target_compile_definitions(${targetName} PRIVATE CUDA_ENABLED)
endif()
if(HIP_ENABLED)
  add_subdirectory(hip)
  target_compile_definitions(${targetName} PRIVATE HIP_ENABLED)
endif()
