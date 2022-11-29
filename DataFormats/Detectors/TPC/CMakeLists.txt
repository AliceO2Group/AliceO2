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

# Comments
#
# * FIXME: ClusterNative depends on SimulationDataFormats : quite annoying for a
#   (supposedly real data component)
#
# * GPUCommonDef -> GPUCommonDef to follow more closely the rest of the AliceO2
#   code, but probably the name should be simplified to remove some common ...

o2_add_library(
  DataFormatsTPC
  SOURCES src/Helpers.cxx
          src/TrackTPC.cxx
          src/LaserTrack.cxx
          src/TPCSectorHeader.cxx
          src/ClusterNativeHelper.cxx
          src/WorkflowHelper.cxx
          src/CompressedClusters.cxx
          src/TrackCuts.cxx
          src/CalibdEdxCorrection.cxx
          src/DCS.cxx
  PUBLIC_LINK_LIBRARIES O2::GPUCommon
                        O2::SimulationDataFormat
                        O2::ReconstructionDataFormats
                        O2::CommonDataFormat
                        O2::Headers
                        O2::Algorithm)

o2_target_root_dictionary(
  DataFormatsTPC
  HEADERS include/DataFormatsTPC/ClusterGroupAttribute.h
          include/DataFormatsTPC/ClusterNative.h
          include/DataFormatsTPC/ClusterNativeHelper.h
          include/DataFormatsTPC/ClusterHardware.h
          include/DataFormatsTPC/Digit.h
          include/DataFormatsTPC/DCS.h
          include/DataFormatsTPC/Helpers.h
          include/DataFormatsTPC/KrCluster.h
          include/DataFormatsTPC/TrackTPC.h
          include/DataFormatsTPC/LaserTrack.h
          include/DataFormatsTPC/Constants.h
          include/DataFormatsTPC/RawDataTypes.h
          include/DataFormatsTPC/Defs.h
          include/DataFormatsTPC/dEdxInfo.h
          include/DataFormatsTPC/CompressedClusters.h
          include/DataFormatsTPC/CTF.h
          include/DataFormatsTPC/IDC.h
          include/DataFormatsTPC/SAC.h
          include/DataFormatsTPC/ZeroSuppression.h
          include/DataFormatsTPC/ZeroSuppressionLinkBased.h
          include/DataFormatsTPC/TrackCuts.h
          include/DataFormatsTPC/LtrCalibData.h
          include/DataFormatsTPC/VDriftCorrFact.h
          include/DataFormatsTPC/CalibdEdxCorrection.h
          include/DataFormatsTPC/BetheBlochAleph.h)

o2_add_test(
  ClusterNative
  SOURCES test/testClusterNative.cxx
  COMPONENT_NAME DataFormats-TPC
  PUBLIC_LINK_LIBRARIES O2::DataFormatsTPC
  LABELS tpc dataformats)

o2_add_test(
  ClusterHardware
  SOURCES test/testClusterHardware.cxx
  COMPONENT_NAME DataFormats-TPC
  PUBLIC_LINK_LIBRARIES O2::DataFormatsTPC
  LABELS tpc dataformats)

o2_add_test(
  CompressedClusters
  SOURCES test/testCompressedClusters.cxx
  COMPONENT_NAME DataFormats-TPC
  PUBLIC_LINK_LIBRARIES O2::DataFormatsTPC
  LABELS tpc dataformats)

o2_add_test(
  TrackCuts
  SOURCES test/test_TrackCuts.cxx
  COMPONENT_NAME DataFormats-TPC
  PUBLIC_LINK_LIBRARIES O2::DataFormatsTPC
  LABELS tpc dataformats)
