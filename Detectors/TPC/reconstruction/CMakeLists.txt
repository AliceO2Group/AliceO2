# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TPCReconstruction
               SOURCES src/AdcClockMonitor.cxx
                       src/ClustererTask.cxx
                       src/GBTFrame.cxx
                       src/GBTFrameContainer.cxx
                       src/HalfSAMPAData.cxx
                       src/HwClusterer.cxx
                       src/RawReader.cxx
                       src/RawReaderCRU.cxx
                       src/RawReaderEventSync.cxx
                       src/SyncPatternMonitor.cxx
                       src/GPUCATracking.cxx
                       src/HardwareClusterDecoder.cxx
                       src/DigitalCurrentClusterIntegrator.cxx
                       src/TPCFastTransformHelperO2.cxx
               PUBLIC_LINK_LIBRARIES FairRoot::Base O2::SimulationDataFormat
                                     O2::TPCBase O2::GPUTracking)

o2_target_root_dictionary(
  TPCReconstruction
  HEADERS include/TPCReconstruction/AdcClockMonitor.h
          include/TPCReconstruction/Clusterer.h
          include/TPCReconstruction/ClustererTask.h
          include/TPCReconstruction/GBTFrame.h
          include/TPCReconstruction/GBTFrameContainer.h
          include/TPCReconstruction/HalfSAMPAData.h
          include/TPCReconstruction/HwClusterer.h
          include/TPCReconstruction/RawReader.h
          include/TPCReconstruction/RawReaderCRU.h
          include/TPCReconstruction/RawReaderEventSync.h
          include/TPCReconstruction/SyncPatternMonitor.h
          include/TPCReconstruction/GPUCATracking.h
          include/TPCReconstruction/HardwareClusterDecoder.h
          include/TPCReconstruction/DigitalCurrentClusterIntegrator.h
          include/TPCReconstruction/TPCFastTransformHelperO2.h)

o2_add_executable(read-gbtframes
                  COMPONENT_NAME tpc
                  PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
                  SOURCES run/readGBTFrames.cxx)

o2_add_executable(read-rawdata
                  COMPONENT_NAME tpc
                  PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
                  SOURCES run/readRawData.cxx)

o2_add_executable(rawreader-cru
                  COMPONENT_NAME tpc
                  PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
                  SOURCES run/rawReaderCRU.cxx)

o2_add_test(SyncPatternMonitor
            COMPONENT_NAME tpc
            LABELS tpc
            PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
            SOURCES test/testTPCSyncPatternMonitor.cxx)

o2_add_test(AdcClockMonitor
            COMPONENT_NAME tpc
            LABELS tpc
            PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
            SOURCES test/testTPCAdcClockMonitor.cxx)

o2_add_test(GPUCATracking
            COMPONENT_NAME tpc
            LABELS tpc
            PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
            SOURCES test/testGPUCATracking.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(HwClusterer
            COMPONENT_NAME tpc
            LABELS tpc
            PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
            SOURCES test/testTPCHwClusterer.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

# The FastTransform  test seems really slow in Debug mode, so use it only in
# release mode (use CONFIGURATIONS keyword)
o2_add_test(FastTransform
            COMPONENT_NAME tpc
            LABELS tpc
            PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
            SOURCES test/testTPCFastTransform.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage CONFIGURATIONS
                        RelWithDebInfo)

# FIXME: should be moved to TPCCalibration as it requires O2::TPCCalibration
# which is built after TPCReconstruction
# o2_add_test_root_macro(macro/RawClusterFinder.C PUBLIC_LINK_LIBRARIES
# O2::DataFormatsTPC O2::TPCBase O2::TPCCalibration O2::TPCReconstruction)

o2_add_test_root_macro(macro/readClusters.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsTPC
                                             O2::SimulationDataFormat
                                             O2::TPCBase
                       LABELS tpc)
o2_add_test_root_macro(macro/testRawRead.C
                       PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
                       LABELS tpc)

# FIXME: should be moved to TPCSimulation as it requires O2::TPCSimulation
# target which is built after reconstruction
# o2_add_test_root_macro(macro/testTracks.C PUBLIC_LINK_LIBRARIES
# ReconstructionDataFormats O2::TPCBase O2::DataFormatsTPC O2::TPCSimulation)
