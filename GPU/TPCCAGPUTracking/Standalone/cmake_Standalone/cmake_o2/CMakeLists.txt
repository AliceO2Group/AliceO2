cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

set(MODULE O2TPCCATracking)

add_library(${MODULE} SHARED
        tracking-ca/AliHLTTPCCATrack.cxx 
        tracking-ca/AliHLTTPCCATrackParam.cxx
        tracking-ca/AliHLTTPCCATracklet.cxx
        tracking-ca/AliHLTTPCCAStartHitsFinder.cxx
        tracking-ca/AliHLTTPCCANeighboursCleaner.cxx
        tracking-ca/AliHLTTPCCAParam.cxx
        tracking-ca/AliHLTTPCCATracker.cxx
        tracking-ca/AliHLTTPCCATrackerFramework.cxx
        tracking-ca/AliHLTTPCCASliceData.cxx
        tracking-ca/AliHLTTPCCASliceOutput.cxx
        tracking-ca/AliHLTTPCCAStandaloneFramework.cxx
        tracking-ca/AliHLTTPCCATrackletConstructor.cxx
        tracking-ca/AliHLTTPCCANeighboursFinder.cxx
        tracking-ca/AliHLTTPCCAGrid.cxx
        tracking-ca/AliHLTTPCCATrackletSelector.cxx
        tracking-ca/AliHLTTPCCAHitArea.cxx
        tracking-ca/AliHLTTPCCAMCPoint.cxx
        tracking-ca/AliHLTTPCCAClusterData.cxx
        tracking-ca/AliHLTTPCCARow.cxx
        tracking-ca/AliHLTTPCCAGPUTracker.cxx
        merger-ca/AliHLTTPCGMMerger.cxx
        merger-ca/AliHLTTPCGMSliceTrack.cxx
        merger-ca/AliHLTTPCGMTrackParam.cxx
        merger-ca/AliHLTTPCGMPhysicalTrackModel.cxx
        merger-ca/AliHLTTPCGMPropagator.cxx
        merger-ca/AliHLTTPCGMPolynomialField.cxx
        merger-ca/AliHLTTPCGMPolynomialFieldCreator.cxx
        interface/AliHLTTPCCAO2Interface.cxx
        interface/AliHLTLogging.cxx
        cmodules/timer.cpp)

add_definitions(-DHLTCA_STANDALONE -DHLTCA_TPC_GEOMETRY_O2 -DHLTCA_BUILD_O2_LIB)

include_directories(include tracking-ca merger-ca)

install(TARGETS ${MODULE}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
