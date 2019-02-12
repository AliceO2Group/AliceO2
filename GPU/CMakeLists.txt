cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(AliGPU NONE)
set(MODULE AliGPU)

install(DIRECTORY
        GPUTracking
        TPCFastTransformation
        TPCSpaceChargeBase
        Common
        cmake
        DESTINATION sources
        PATTERN "GPUTracking/utils/.*" EXCLUDE
)
