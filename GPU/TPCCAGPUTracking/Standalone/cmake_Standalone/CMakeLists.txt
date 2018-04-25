cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

set(MODULE O2TPCCATracking)

add_library(${MODULE} SHARED
        ../code/AliHLTTPCCATrack.cxx 
        ../code/AliHLTTPCCATrackParam.cxx
        ../code/AliHLTTPCCATracklet.cxx
        ../code/AliHLTTPCCAStartHitsFinder.cxx
        ../code/AliHLTTPCCANeighboursCleaner.cxx
        ../code/AliHLTTPCCAParam.cxx
        ../code/AliHLTTPCCATracker.cxx
        ../code/AliHLTTPCCATrackerFramework.cxx
        ../code/AliHLTTPCCASliceData.cxx
        ../code/AliHLTTPCCASliceOutput.cxx
        ../code/AliHLTTPCCAStandaloneFramework.cxx
        ../code/AliHLTTPCCATrackletConstructor.cxx
        ../code/AliHLTTPCCANeighboursFinder.cxx
        ../code/AliHLTTPCCAGrid.cxx
        ../code/AliHLTTPCCATrackletSelector.cxx
        ../code/AliHLTTPCCAHitArea.cxx
        ../code/AliHLTTPCCAMCPoint.cxx
        ../code/AliHLTTPCCAClusterData.cxx
        ../code/AliHLTTPCCARow.cxx
        ../code/AliHLTTPCCAGPUTracker.cxx
        ../merger-ca/AliHLTTPCGMMerger.cxx
        ../merger-ca/AliHLTTPCGMSliceTrack.cxx
        ../merger-ca/AliHLTTPCGMTrackParam.cxx
        ../merger-ca/AliHLTTPCGMPhysicalTrackModel.cxx
        ../merger-ca/AliHLTTPCGMPropagator.cxx
        ../merger-ca/AliHLTTPCGMPolynomialField.cxx
        ../merger-ca/AliHLTTPCGMPolynomialFieldCreator.cxx
        ../interface/AliHLTTPCCAO2Interface.cxx
        ../interface/AliHLTLogging.cxx
        ../cmodules/timer.cpp)

#set_target_properties(${MODULE} PROPERTIES SUFFIX .so) #Could force .so on MacOS, but no longer needed

add_definitions(-DHLTCA_STANDALONE -DHLTCA_TPC_GEOMETRY_O2 -DHLTCA_BUILD_O2_LIB)

include_directories(../include ../code ../merger-ca ../cagpubuild)

install(TARGETS ${MODULE}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(FILES
        ../interface/AliHLTTPCCAO2Interface.h
        ../code/AliHLTTPCCAClusterData.h
        ../code/AliHLTTPCCAMath.h
        ../code/AliHLTTPCCADef.h
        ../code/AliHLTTPCCASettings.h
        ../merger-ca/AliHLTTPCGMMergedTrack.h
        ../merger-ca/AliHLTTPCGMMergedTrackHit.h
        ../merger-ca/AliHLTTPCGMTrackParam.h
        DESTINATION include)

install(DIRECTORY
        ../../tracking-ca
        ../../merger-ca
        ../interface
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
