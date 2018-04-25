cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

set(MODULE O2TPCCATracking)

add_library(${MODULE} SHARED
        ../SliceTracker/AliHLTTPCCATrack.cxx 
        ../SliceTracker/AliHLTTPCCATrackParam.cxx
        ../SliceTracker/AliHLTTPCCATracklet.cxx
        ../SliceTracker/AliHLTTPCCAStartHitsFinder.cxx
        ../SliceTracker/AliHLTTPCCANeighboursCleaner.cxx
        ../SliceTracker/AliHLTTPCCAParam.cxx
        ../SliceTracker/AliHLTTPCCATracker.cxx
        ../SliceTracker/AliHLTTPCCATrackerFramework.cxx
        ../SliceTracker/AliHLTTPCCASliceData.cxx
        ../SliceTracker/AliHLTTPCCASliceOutput.cxx
        ../SliceTracker/AliHLTTPCCAStandaloneFramework.cxx
        ../SliceTracker/AliHLTTPCCATrackletConstructor.cxx
        ../SliceTracker/AliHLTTPCCANeighboursFinder.cxx
        ../SliceTracker/AliHLTTPCCAGrid.cxx
        ../SliceTracker/AliHLTTPCCATrackletSelector.cxx
        ../SliceTracker/AliHLTTPCCAHitArea.cxx
        ../SliceTracker/AliHLTTPCCAMCPoint.cxx
        ../SliceTracker/AliHLTTPCCAClusterData.cxx
        ../SliceTracker/AliHLTTPCCARow.cxx
        ../SliceTracker/AliHLTTPCCAGPUTracker.cxx
        ../Merger/AliHLTTPCGMMerger.cxx
        ../Merger/AliHLTTPCGMSliceTrack.cxx
        ../Merger/AliHLTTPCGMTrackParam.cxx
        ../Merger/AliHLTTPCGMPhysicalTrackModel.cxx
        ../Merger/AliHLTTPCGMPropagator.cxx
        ../Merger/AliHLTTPCGMPolynomialField.cxx
        ../Merger/AliHLTTPCGMPolynomialFieldCreator.cxx
        ../Interface/AliHLTTPCCAO2Interface.cxx
        ../Interface/AliHLTLogging.cxx
        ../cmodules/timer.cpp)

#set_target_properties(${MODULE} PROPERTIES SUFFIX .so) #Could force .so on MacOS, but no longer needed

add_definitions(-DHLTCA_STANDALONE -DHLTCA_TPC_GEOMETRY_O2 -DHLTCA_BUILD_O2_LIB)

include_directories(../include ../SliceTracker ../Merger ../GlobalTracker)

install(TARGETS ${MODULE}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(FILES
        ../Interface/AliHLTTPCCAO2Interface.h
        ../SliceTracker/AliHLTTPCCAClusterData.h
        ../SliceTracker/AliHLTTPCCAMath.h
        ../SliceTracker/AliHLTTPCCADef.h
        ../SliceTracker/AliHLTTPCCASettings.h
        ../Merger/AliHLTTPCGMMergedTrack.h
        ../Merger/AliHLTTPCGMMergedTrackHit.h
        ../Merger/AliHLTTPCGMTrackParam.h
        DESTINATION include)

install(DIRECTORY
        ../../SliceTracker
        ../../Merger
        ../Interface
        ../include
        ../cmodules
        DESTINATION source
        PATTERN "AliHLTTPCClusterMCData.h" EXCLUDE
        PATTERN "AliHLTArray.h" EXCLUDE
)
install(FILES
        cmake_o2/CMakeLists.txt
        DESTINATION source)
install(FILES
        ../../../BASE/AliHLTArray.h
        ../../AliHLTTPCClusterMCData.h
        DESTINATION source/include)
