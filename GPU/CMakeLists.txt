cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(AliTPCCommon NONE)
set(MODULE AliTPCCommon)

install(DIRECTORY
        TPCCAGPUTracking
        TPCFastTransformation
        TPCSpaceChargeBase
        Common
        cmake
        DESTINATION sources
        PATTERN "TPCCAGPUTracking/Standalone/cmodules/.*" EXCLUDE
)
