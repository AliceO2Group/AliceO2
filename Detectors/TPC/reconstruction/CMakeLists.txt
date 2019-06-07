o2_add_library(TPCReconstruction
               SOURCES
               src/AdcClockMonitor.cxx
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
               PUBLIC_LINK_LIBRARIES
               FairRoot::Base
               O2::SimulationDataFormat
               O2::TPCBase
               O2::GPUTracking)

o2_target_root_dictionary(
  TPCReconstruction
  HEADERS
  include/TPCReconstruction/AdcClockMonitor.h
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
                  COMPONENT_NAME
                  tpc
                  PUBLIC_LINK_LIBRARIES
                  O2::TPCReconstruction
                  SOURCES
                  run/readGBTFrames.cxx)
o2_add_executable(read-rawdata
                  COMPONENT_NAME
                  tpc
                  PUBLIC_LINK_LIBRARIES
                  O2::TPCReconstruction
                  SOURCES
                  run/readRawData.cxx)
o2_add_executable(rawreader-cru
                  COMPONENT_NAME
                  tpc
                  PUBLIC_LINK_LIBRARIES
                  O2::TPCReconstruction
                  SOURCES
                  run/rawReaderCRU.cxx)

o2_add_test(SyncPatternMonitor
            COMPONENT_NAME
            tpc
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCReconstruction
            SOURCES
            test/testTPCSyncPatternMonitor.cxx)
o2_add_test(AdcClockMonitor
            COMPONENT_NAME
            tpc
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCReconstruction
            SOURCES
            test/testTPCAdcClockMonitor.cxx)
o2_add_test(GPUCATracking
            COMPONENT_NAME
            tpc
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCReconstruction
            SOURCES
            test/testGPUCATracking.cxx)
o2_add_test(HwClusterer
            COMPONENT_NAME
            tpc
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCReconstruction
            SOURCES
            test/testTPCHwClusterer.cxx)
o2_add_test(FastTransform
            COMPONENT_NAME
            tpc
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCReconstruction
            SOURCES
            test/testTPCFastTransform.cxx)

# FIXME: should be moved to TPCCalibration as it requires O2::TPCCalibration
# which is built after TPCReconstruction
# o2_add_test_root_macro(macro/RawClusterFinder.C PUBLIC_LINK_LIBRARIES
# O2::DataFormatsTPC O2::TPCBase O2::TPCCalibration O2::TPCReconstruction)
o2_add_test_root_macro(macro/convertTracks.C PUBLIC_LINK_LIBRARIES
                       O2::DataFormatsTPC O2::ReconstructionDataFormats)
o2_add_test_root_macro(macro/dEdxRes.C PUBLIC_LINK_LIBRARIES O2::DataFormatsTPC
                       O2::TPCBase)
o2_add_test_root_macro(macro/readClusterMCtruth.C PUBLIC_LINK_LIBRARIES
                       O2::DataFormatsTPC O2::SimulationDataFormat)
o2_add_test_root_macro(macro/readClusters.C PUBLIC_LINK_LIBRARIES
                       O2::DataFormatsTPC O2::SimulationDataFormat O2::TPCBase)
# FIXME: should be moved to TPCCalibration as it requires O2::TPCCalibration
# which is built after TPCReconstruction
# o2_add_test_root_macro(macro/runRawClusterFinder.C PUBLIC_LINK_LIBRARIES
# O2::DataFormatsTPC O2::TPCBase O2::TPCCalibration O2::TPCReconstruction)
o2_add_test_root_macro(macro/testClustererData.C PUBLIC_LINK_LIBRARIES
                       O2::DataFormatsTPC O2::TPCBase O2::TPCReconstruction)
o2_add_test_root_macro(macro/testRawRead.C PUBLIC_LINK_LIBRARIES
                       O2::TPCReconstruction)
# FIXME: should be moved to TPCSimulation as it requires O2::TPCSimulation
# target which is built after reconstruction
# o2_add_test_root_macro(macro/testTracks.C PUBLIC_LINK_LIBRARIES
# ReconstructionDataFormats O2::TPCBase O2::DataFormatsTPC O2::TPCSimulation)
