set(MODULE O2TPCCATracking)

add_library(${MODULE} SHARED code/AliHLTTPCCATrack.cxx 
                                                                code/AliHLTTPCCATrackParam.cxx
                                                                code/AliHLTTPCCATracklet.cxx
                                                                code/AliHLTTPCCAStartHitsFinder.cxx
                                                                code/AliHLTTPCCANeighboursCleaner.cxx
                                                                code/AliHLTTPCCAParam.cxx
                                                                code/AliHLTTPCCATracker.cxx
                                                                code/AliHLTTPCCATrackerFramework.cxx
                                                                code/AliHLTTPCCASliceData.cxx
                                                                code/AliHLTTPCCASliceOutput.cxx
                                                                code/AliHLTTPCCAStandaloneFramework.cxx
                                                                code/AliHLTTPCCATrackletConstructor.cxx
                                                                code/AliHLTTPCCANeighboursFinder.cxx
                                                                code/AliHLTTPCCAGrid.cxx
                                                                code/AliHLTTPCCATrackletSelector.cxx
                                                                code/AliHLTTPCCAHitArea.cxx
                                                                code/AliHLTTPCCAMCPoint.cxx
                                                                code/AliHLTTPCCAClusterData.cxx
                                                                code/AliHLTTPCCARow.cxx
                                                                code/AliHLTTPCCAGPUTracker.cxx
                                                                merger-ca/AliHLTTPCGMMerger.cxx
                                                                merger-ca/AliHLTTPCGMSliceTrack.cxx
                                                                merger-ca/AliHLTTPCGMTrackParam.cxx
                                                                interface/AliHLTTPCCAO2Interface.cxx
                                                                standalone/AliHLTLogging.cxx)

add_definitions(-DHLTCA_STANDALONE -DHLTCA_TPC_GEOMETRY_O2)

include_directories(include code base merger-ca cagpubuild)

install(TARGETS ${MODULE}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(FILES interface/AliHLTTPCCAO2Interface.h DESTINATION include)
