cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

include(../cmake_AliRoot/CMakeLists.txt)

install(FILES
        ../code/AliHLTTPCCAClusterData.h
        ../code/AliHLTTPCCAMath.h
        ../code/AliHLTTPCCADef.h
        ../merger-ca/AliHLTTPCGMMergedTrack.h
        ../merger-ca/AliHLTTPCGMTrackParam.h
DESTINATION include)
