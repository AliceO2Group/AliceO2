cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

set(MODULE AliTPCCommon)

install(DIRECTORY
        TPCCAGPUTracking
        TPCFastTransformation
        DESTINATION sources
        PATTERN "TPCCAGPUTracking/Standalone/cmodules/.*" EXCLUDE
)
