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

o2_add_library(TPCInterpolationWorkflow
               SOURCES src/TPCInterpolationSpec.cxx
                       src/TPCResidualWriterSpec.cxx
                       src/TPCResidualReaderSpec.cxx
                       src/TPCUnbinnedResidualReaderSpec.cxx
                       include/TPCInterpolationWorkflow/TPCResidualAggregatorSpec.h
               PUBLIC_LINK_LIBRARIES O2::ITSWorkflow
                                     O2::SpacePoints
                                     O2::GlobalTrackingWorkflow
                                     O2::TOFWorkflowIO
                                     O2::Framework
                                     O2::TRDWorkflowIO
                                     )

o2_add_executable(scdcalib-interpolation-workflow
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-interpolation-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCInterpolationWorkflow)

o2_add_executable(residual-aggregator
                  COMPONENT_NAME calibration
                  SOURCES src/tpc-residual-aggregator.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsCalibration O2::TPCInterpolationWorkflow)

o2_add_executable(static-map-creator
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-map-creator.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCInterpolationWorkflow)
