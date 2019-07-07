# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(MODULE GPUTracking)

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  if(ENABLE_CUDA OR ENABLE_OPENCL1 OR ENABLE_OPENCL2 OR ENABLE_HIP)
    cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
    find_package(O2GPU)
  endif()
else()
  if(OPENGL_FOUND
     AND GLFW_FOUND
     AND GLEW_FOUND
     AND OPENGL_GLU_FOUND
     AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(GPUCA_EVENT_DISPLAY ON)
  endif()
  set(GPUCA_QA ON)
endif()

set(SRCS
    SliceTracker/GPUTPCTrack.cxx
    SliceTracker/GPUTPCBaseTrackParam.cxx
    SliceTracker/GPUTPCTrackParam.cxx
    SliceTracker/GPUTPCTracklet.cxx
    SliceTracker/GPUTPCStartHitsFinder.cxx
    SliceTracker/GPUTPCStartHitsSorter.cxx
    SliceTracker/GPUTPCNeighboursCleaner.cxx
    SliceTracker/GPUTPCTracker.cxx
    SliceTracker/GPUTPCSliceData.cxx
    SliceTracker/GPUTPCSliceOutput.cxx
    SliceTracker/GPUTPCTrackletConstructor.cxx
    SliceTracker/GPUTPCNeighboursFinder.cxx
    SliceTracker/GPUTPCGrid.cxx
    SliceTracker/GPUTPCTrackletSelector.cxx
    SliceTracker/GPUTPCHitArea.cxx
    SliceTracker/GPUTPCClusterData.cxx
    SliceTracker/GPUTPCRow.cxx
    Merger/GPUTPCGMMerger.cxx
    Merger/GPUTPCGMSliceTrack.cxx
    Merger/GPUTPCGMTrackParam.cxx
    Merger/GPUTPCGMPropagator.cxx
    Merger/GPUTPCGMPhysicalTrackModel.cxx
    Merger/GPUTPCGMPolynomialField.cxx
    Merger/GPUTPCGMPolynomialFieldManager.cxx
    Merger/GPUTPCGMMergerGPU.cxx
    TRDTracking/GPUTRDTrack.cxx
    TRDTracking/GPUTRDTracker.cxx
    TRDTracking/GPUTRDTrackletWord.cxx
    TRDTracking/GPUTRDTrackerGPU.cxx
    Base/GPUParam.cxx)

set(SRCS_NO_CINT
    Base/GPUDataTypes.cxx
    Base/GPUReconstruction.cxx
    Base/GPUReconstructionCPU.cxx
    Base/GPUSettings.cxx
    Base/GPUProcessor.cxx
    Base/GPUMemoryResource.cxx
    Base/GPUGeneralKernels.cxx
    Base/GPUReconstructionDeviceBase.cxx
    Base/GPUReconstructionConvert.cxx
    Global/GPUChain.cxx
    Global/GPUChainTracking.cxx
    Standalone/utils/timer.cpp)

set(SRCS_NO_H SliceTracker/GPUTPCTrackerDump.cxx
              Global/GPUChainTrackingDebugAndProfiling.cxx)

set(HDRS_INSTALL
    SliceTracker/GPUTPCMCInfo.h
    SliceTracker/GPUTPCHit.h
    SliceTracker/GPUTPCHitId.h
    SliceTracker/GPUTPCSliceOutCluster.h
    SliceTracker/GPUTPCSliceOutTrack.h
    SliceTracker/GPUTPCTrackLinearisation.h
    Merger/GPUTPCGMBorderTrack.h
    Merger/GPUTPCGMOfflineStatisticalErrors.h
    Merger/GPUTPCGMMergedTrack.h
    Merger/GPUTPCGMMergedTrackHit.h
    TRDTracking/GPUTRDDef.h
    TRDTracking/GPUTRDTrackPoint.h
    TRDTracking/GPUTRDTrack.h
    TRDTracking/GPUTRDTrackData.h
    TRDTracking/GPUTRDTrackPoint.h
    TRDTracking/GPUTRDTrackletLabels.h
    Base/GPUReconstructionIncludes.h
    SliceTracker/GPUTPCDef.h
    SliceTracker/GPUTPCGeometry.h
    SliceTracker/GPULogging.h
    SliceTracker/ClusterNativeAccessExt.h
    TRDTracking/GPUTRDGeometry.h
    TRDTracking/GPUTRDTrackerDebug.h
    TRDTracking/GPUTRDInterfaces.h
    Standalone/display/GPUDisplayConfig.h
    Standalone/qa/GPUQAConfig.h
    Base/GPUConstantMem.h
    Base/GPUReconstructionKernels.h
    Base/GPUReconstructionIncludesITS.h
    Base/GPUReconstructionHelpers.h
    Base/GPUOutputControl.h)

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  set(SRCS
      ${SRCS}
      Merger/GPUTPCGlobalMergerComponent.cxx
      SliceTracker/GPUTPCTrackerComponent.cxx
      Merger/GPUTPCGMTracksToTPCSeeds.cxx
      DataCompression/AliHLTTPCClusterStatComponent.cxx
      TRDTracking/GPUTRDTrackerComponent.cxx
      TRDTracking/GPUTRDTrackletReaderComponent.cxx
      Global/AliHLTGPUDumpComponent.cxx)

  set(SRCS_NO_CINT ${SRCS_NO_CINT}
                   ${CMAKE_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCGeometry.cxx
                   ${CMAKE_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCLog.cxx
                   ${CMAKE_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCDefinitions.cxx
                   ${CMAKE_SOURCE_DIR}/HLT/TRD/AliHLTTRDDefinitions.cxx)

  set(HDRS_INSTALL
      ${HDRS_INSTALL}
      SliceTracker/GPUTPCDefinitions.h
      ITS/GPUITSFitter.h
      ITS/GPUITSFitterKernels.h
      ITS/GPUITSTrack.h
      TPCConvert/GPUTPCConvert.h
      TPCConvert/GPUTPCConvertKernel.h
      DataCompression/GPUTPCCompression.h
      DataCompression/GPUTPCCompressionTrackModel.h
      DataCompression/GPUTPCCompressionKernels.h
      DataCompression/TPCClusterDecompressor.h
      DataCompression/GPUTPCClusterStatistics.h)
else()
  set(SRCS_NO_CINT
      ${SRCS_NO_CINT}
      Standalone/display/GPUDisplayBackend.cpp
      Global/GPUChainITS.cxx
      ITS/GPUITSFitter.cxx
      ITS/GPUITSFitterKernels.cxx
      dEdx/GPUdEdx.cxx
      TPCConvert/GPUTPCConvert.cxx
      TPCConvert/GPUTPCConvertKernel.cxx
      DataCompression/GPUTPCCompression.cxx
      DataCompression/GPUTPCCompressionTrackModel.cxx
      DataCompression/GPUTPCCompressionKernels.cxx
      DataCompression/TPCClusterDecompressor.cxx
      DataCompression/GPUTPCClusterStatistics.cxx)
  set(HDRS_INSTALL ${HDRS_INSTALL} Interface/GPUO2InterfaceConfiguration.h
                   ITS/GPUITSTrack.h dEdx/GPUdEdxInfo.h)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  set(SRCS ${SRCS} Interface/GPUO2Interface.cxx)
  set(HDRS_CINT_O2 ${HDRS_CINT_O2} Interface/GPUO2Interface.h)
endif()

if(GPUCA_EVENT_DISPLAY)
  set(SRCS_NO_CINT ${SRCS_NO_CINT} Standalone/utils/qsem.cpp
                   Standalone/display/GPUDisplay.cpp
                   Standalone/display/GPUDisplayBackendGlfw.cpp)
  set(SRCS_NO_H ${SRCS_NO_H} Standalone/display/GPUDisplayQuaternion.cpp
                Standalone/display/GPUDisplayInterpolation.cpp
                Standalone/display/GPUDisplayKeys.cpp)
else()
  set(HDRS_INSTALL ${HDRS_INSTALL} Standalone/display/GPUDisplay.h)
endif()

if(GPUCA_QA)
  set(SRCS_NO_CINT ${SRCS_NO_CINT} Standalone/qa/GPUQA.cpp)
else()
  set(HDRS_INSTALL ${HDRS_INSTALL} Standalone/qa/GPUQA.h)
endif()

if(GPUCA_EVENT_DISPLAY OR GPUCA_QA)
  set(HDRS_INSTALL ${HDRS_INSTALL} Standalone/qconfigoptions.h)
endif()

string(REPLACE ".cxx" ".h" HDRS_CINT_ALIROOT "${SRCS}")
string(REPLACE ".cxx" ".h" HDRS_TMP "${SRCS_NO_CINT}")
set(HDRS_INSTALL ${HDRS_INSTALL} ${HDRS_TMP})
unset(HDRS_TMP)

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 TARGETVARNAME targetName
                 PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                       O2::DataFormatsTPC
                                       O2::TRDBase
                                       O2::ITStracking
                                       O2::TPCFastTransformation
                                       O2::DebugGUI
                 PUBLIC_INCLUDE_DIRECTORIES SliceTracker
                                            Base
                                            TPCConvert
                                            dEdx
                                            ITS
                                            TRDTracking
                                            Standalone
                                            Standalone/qa
                                            Standalone/display
                                            Global
                                            HLTHeaders
                                            Interface
                                            Merger
                                            DataCompression
                 TARGETVARNAME targetName
                 SOURCES ${SRCS} ${SRCS_NO_CINT} ${SRCS_NO_H})

  target_include_directories(
    ${targetName}
    PRIVATE $<TARGET_PROPERTY:O2::Framework,INTERFACE_INCLUDE_DIRECTORIES>)

  o2_target_root_dictionary(${MODULE}
                            HEADERS ${HDRS_CINT_O2}
                            LINKDEF GPUTrackingLinkDef_O2.h)

  o2_add_test(${MODULE}
              PUBLIC_LINK_LIBRARIES O2::${MODULE}
              SOURCES ctest/testGPUTracking.cxx
              COMPONENT_NAME GPU
              LABELS gpu)

  target_compile_definitions(${targetName} PRIVATE GPUCA_O2_LIB
                             GPUCA_TPC_GEOMETRY_O2 HAVE_O2HEADERS)

  install(FILES ${HDRS_CINT_ALIROOT} ${HDRS_CINT_O2} ${HDRS_INSTALL}
          DESTINATION include/GPU)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  add_definitions(-DGPUCA_ALIROOT_LIB)
  include_directories(Standalone/display Standalone/qa)
  include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
  include_directories(${CMAKE_SOURCE_DIR}/HLT/BASE
                      ${CMAKE_SOURCE_DIR}/HLT/BASE/util
                      ${CMAKE_SOURCE_DIR}/HLT/TPCLib
                      ${CMAKE_SOURCE_DIR}/HLT/TPCLib/transform
                      ${CMAKE_SOURCE_DIR}/HLT/TPCLib/comp
                      ${CMAKE_SOURCE_DIR}/HLT/TRD
                      ${CMAKE_SOURCE_DIR}/TRD/TRDbase
                      ${CMAKE_SOURCE_DIR}/STEER/STEERBase
                      ${CMAKE_SOURCE_DIR}/STEER/STEER
                      ${CMAKE_SOURCE_DIR}/STEER/ESD
                      ${CMAKE_SOURCE_DIR}/STEER/CDB
                      ${CMAKE_SOURCE_DIR}/TPC/TPCbase
                      ${CMAKE_SOURCE_DIR}/TPC/TPCcalib
                      ${CMAKE_SOURCE_DIR}/TPC/TPCrec
                      ${CMAKE_SOURCE_DIR}/GPU/Common
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/SliceTracker
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Merger
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Global
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/dEdx
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/ITS
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/TPCConvert
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/TRDTracking
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Standalone
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base/cuda
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base/hip
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base/opencl
                      ${CMAKE_SOURCE_DIR}/GPU/TPCFastTransformation)
  alice_usevc()

  # Generate the dictionary
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  generate_dictionary_flat("Ali${MODULE}" "GPUTrackingLinkDef_AliRoot.h"
                           "${HDRS_CINT_ALIROOT}" "${incdirs}")

  # Generate the ROOT map Dependecies
  set(LIBDEPS
      STEERBase
      STEER
      ESD
      CDB
      HLTbase
      TPCbase
      TPCrec
      TPCcalib
      TRDbase
      AliTPCFastTransformation)
  generate_rootmap("Ali${MODULE}" "${LIBDEPS}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/GPUTrackingLinkDef_AliRoot.h")
  # Don't pass Vc to root
  set(LIBDEPS ${LIBDEPS} Vc)

  # Add a library to the project using the specified source files
  add_library_tested(Ali${MODULE}
                     SHARED
                     ${SRCS}
                     ${SRCS_NO_CINT}
                     ${SRCS_NO_H}
                     G__Ali${MODULE}.cxx)
  target_link_libraries(Ali${MODULE} ${LIBDEPS})

  # Additional compilation flags
  set_target_properties(Ali${MODULE} PROPERTIES COMPILE_FLAGS "")

  # System dependent: Modify the way the library is build
  if(${CMAKE_SYSTEM} MATCHES Darwin)
    set_target_properties(Ali${MODULE}
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif(${CMAKE_SYSTEM} MATCHES Darwin)

  # Installation
  install(TARGETS Ali${MODULE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)

  install(FILES ${HDRS_CINT_ALIROOT} ${HDRS_INSTALL} DESTINATION include)
  install(DIRECTORY Standalone/utils
          DESTINATION include
          FILES_MATCHING
          PATTERN *.h)

  set(targetName Ali${MODULE})
  add_library(O2::${MODULE} ALIAS Ali${MODULE})
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_library(${MODULE} SHARED ${SRCS} ${SRCS_NO_CINT} ${SRCS_NO_H})
  set(targetName ${MODULE})
  add_library(O2::${MODULE} ALIAS ${MODULE})
endif()

if(GPUCA_EVENT_DISPLAY)
  message(STATUS "Building GPU Event Display")
  target_compile_definitions(${targetName} PRIVATE BUILD_EVENT_DISPLAY)
  target_link_libraries(${targetName}
                        PUBLIC ${GLEW_LIBRARIES} ${GLFW_LIBRARIES} OpenGL::GL
                               OpenGL::GLU)
endif()

if(GPUCA_QA)
  message(STATUS "Building GPU QA")
  target_compile_definitions(${targetName} PRIVATE BUILD_QA)
endif()

if(OpenMP_CXX_FOUND)
  message(STATUS "GPU: Using OpenMP: ${OpenMP_CXX_SPEC_DATE}")
  target_compile_definitions(${targetName} PRIVATE GPUCA_HAVE_OPENMP)
  target_link_libraries(${targetName} PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CUDA_ENABLED)
  target_compile_definitions(${targetName} PRIVATE CUDA_ENABLED)
endif()
if(OPENCL1_ENABLED)
  target_compile_definitions(${targetName} PRIVATE OPENCL1_ENABLED)
endif()
if(OPENCL2_ENABLED)
  target_compile_definitions(${targetName} PRIVATE OPENCL2_ENABLED)
endif()
if(HIP_ENABLED)
  target_compile_definitions(${targetName} PRIVATE HIP_ENABLED)
endif()

if(CUDA_ENABLED OR OPENCL1_ENABLED OR HIP_ENABLED)
  if(CMAKE_SYSTEM_NAME MATCHES Darwin)
    message(WARNING "GPU Tracking disabled on MacOS")
  else()
    if(CUDA_ENABLED)
      add_subdirectory(Base/cuda)
    endif()
    if(OPENCL1_ENABLED)
      add_subdirectory(Base/opencl)
    endif()
    if(HIP_ENABLED)
      add_subdirectory(Base/hip)
    endif()
  endif()
endif()
