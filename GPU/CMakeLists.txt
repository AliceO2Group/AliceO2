cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(AliTPCCommon NONE)
set(MODULE AliTPCCommon)

install(DIRECTORY
        CAGPUTracking
        TPCFastTransformation
        TPCSpaceChargeBase
        Common
        cmake
        DESTINATION sources
        PATTERN "CAGPUTracking/Standalone/cmodules/.*" EXCLUDE
)
