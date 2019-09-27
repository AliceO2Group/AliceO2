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
    include(FeatureSummary)
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
  if(ROOT_FOUND)
    set(GPUCA_QA ON)
  endif()
endif()

# General sources
set(SRCS
    SliceTracker/GPUTPCTrack.cxx
    SliceTracker/GPUTPCTrackParam.cxx
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
    Standalone/utils/timer.cxx)

set(SRCS_NO_H SliceTracker/GPUTPCTrackerDump.cxx
              Global/GPUChainTrackingDebugAndProfiling.cxx)

set(HDRS_INSTALL
    SliceTracker/GPUTPCMCInfo.h
    SliceTracker/GPUTPCHit.h
    SliceTracker/GPUTPCHitId.h
    SliceTracker/GPUTPCClusterData.h
    SliceTracker/GPUTPCBaseTrackParam.h
    SliceTracker/GPUTPCTracklet.h
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
    TRDTracking/GPUTRDGeometry.h
    TRDTracking/GPUTRDTrackerDebug.h
    TRDTracking/GPUTRDInterfaces.h
    Standalone/display/GPUDisplayConfig.h
    Standalone/qa/GPUQAConfig.h
    Base/GPUConstantMem.h
    Base/GPUReconstructionKernels.h
    Base/GPUReconstructionIncludesITS.h
    Base/GPUReconstructionHelpers.h
    Base/GPUOutputControl.h
    Base/GPUOutputControl.h
    Base/GPUO2DataTypes.h
    Base/GPUMemorySizeScalers.h
    dEdx/GPUdEdxInfo.h
    TPCConvert/GPUTPCConvertImpl.h
)

# Sources only for O2
if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  set(SRCS ${SRCS} Interface/GPUO2Interface.cxx)
  set(HDRS_CINT_O2 ${HDRS_CINT_O2} Interface/GPUO2Interface.h)
endif()

# Sources for O2 and for Standalone if requested in config file
if(ALIGPU_BUILD_TYPE STREQUAL "O2" OR ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(SRCS_NO_CINT ${SRCS_NO_CINT} Standalone/display/GPUDisplayBackend.cxx)
endif()
if(ALIGPU_BUILD_TYPE STREQUAL "O2" OR CONFIG_ROOT_AND_O2_EXTENSIONS)
    set(SRCS_NO_CINT
        ${SRCS_NO_CINT}
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
else()
  set(HDRS_INSTALL
      ${HDRS_INSTALL}
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
endif()

# Sources only for AliRoot
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

  set(HDRS_INSTALL ${HDRS_INSTALL} SliceTracker/GPUTPCDefinitions.h)
endif()

# Optional sources depending on optional dependencies
if(GPUCA_EVENT_DISPLAY)
  set(SRCS_NO_CINT ${SRCS_NO_CINT} Standalone/utils/qsem.cxx
                   Standalone/display/GPUDisplay.cxx
                   Standalone/display/GPUDisplayBackendGlfw.cxx)
  set(SRCS_NO_H ${SRCS_NO_H} Standalone/display/GPUDisplayQuaternion.cxx
                Standalone/display/GPUDisplayInterpolation.cxx
                Standalone/display/GPUDisplayKeys.cxx)
else()
  set(HDRS_INSTALL ${HDRS_INSTALL} Standalone/display/GPUDisplay.h)
endif()

if(GPUCA_QA)
  set(SRCS_NO_CINT ${SRCS_NO_CINT} Standalone/qa/GPUQA.cxx)
else()
  set(HDRS_INSTALL ${HDRS_INSTALL} Standalone/qa/GPUQA.h)
endif()

if(GPUCA_EVENT_DISPLAY OR GPUCA_QA)
  set(HDRS_INSTALL ${HDRS_INSTALL} Standalone/qconfigoptions.h)
endif()

# Update HDR variables with files derrived from SRC variables
string(REPLACE ".cxx" ".h" HDRS_CINT_ALIROOT "${SRCS}")
string(REPLACE ".cxx" ".h" HDRS_TMP "${SRCS_NO_CINT}")
set(HDRS_INSTALL ${HDRS_INSTALL} ${HDRS_TMP})
unset(HDRS_TMP)

if (TARGET AliceO2::DebugGUI)
  set(DEBUGGUI_TARGET AliceO2::DebugGUI)
else()
  set(DEBUGGUI_TARGET O2::DebugGUI)
endif()
# Main CMake part for O2
if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 TARGETVARNAME targetName
                 PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                       O2::DataFormatsTPC
                                       O2::TRDBase
                                       O2::ITStracking
                                       O2::TPCFastTransformation
                                       ${DEBUGGUI_TARGET}
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
  install(DIRECTORY Standalone/utils
          DESTINATION include/GPU
          FILES_MATCHING PATTERN *.h)

  o2_add_test_root_macro(Standalone/tools/createGeo.C
                         PUBLIC_LINK_LIBRARIES O2::GPUTracking
                         LABELS its COMPILE_ONLY)
  o2_add_test_root_macro(Standalone/tools/createLUT.C
                         PUBLIC_LINK_LIBRARIES O2::GPUTracking
                         LABELS its COMPILE_ONLY)
endif()

# Main CMake part for AliRoot
if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  set(HDRS_CINT_ALIROOT ${HDRS_CINT_ALIROOT}
                        Base/GPUO2DataTypes.h)
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
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base/opencl-common
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base/opencl
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base/opencl2
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
          FILES_MATCHING PATTERN *.h)

  set(targetName Ali${MODULE})
  add_library(O2::${MODULE} ALIAS Ali${MODULE})
endif()

# Main CMake part for Standalone
if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_library(${MODULE} SHARED ${SRCS} ${SRCS_NO_CINT} ${SRCS_NO_H})
  set(targetName ${MODULE})
  add_library(O2::${MODULE} ALIAS ${MODULE})
endif()

if(GPUCA_EVENT_DISPLAY)
  message(STATUS "Building GPU Event Display")
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY)
  target_link_libraries(${targetName}
                        PUBLIC ${GLEW_LIBRARIES} ${GLFW_LIBRARIES} OpenGL::GL
                               OpenGL::GLU)
endif()

# Add compile definitions and libraries depending on available optional dependencies
if(GPUCA_QA)
  message(STATUS "Building GPU QA")
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_QA)
endif()

if(OpenMP_CXX_FOUND)
  message(STATUS "GPU: Using OpenMP: ${OpenMP_CXX_SPEC_DATE}")
  target_compile_definitions(${targetName} PRIVATE GPUCA_HAVE_OPENMP)
  # Must be private, depending libraries might be compiled by compiler not understanding -fopenmp
  target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
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

# Add CMake recipes for GPU Tracking librararies
if(CUDA_ENABLED OR OPENCL1_ENABLED OR OPENCL2_ENABLED OR HIP_ENABLED)
  if(CMAKE_SYSTEM_NAME MATCHES Darwin)
    message(WARNING "GPU Tracking disabled on MacOS")
  else()
    if(CUDA_ENABLED)
      add_subdirectory(Base/cuda)
    endif()
    if(OPENCL1_ENABLED OR OPENCL2_ENABLED)
      add_subdirectory(Base/opencl-common)
    endif()
    if(OPENCL1_ENABLED)
      add_subdirectory(Base/opencl)
    endif()
    if(OPENCL2_ENABLED)
      add_subdirectory(Base/opencl2)
    endif()
    if(HIP_ENABLED)
      add_subdirectory(Base/hip)
    endif()
  endif()
endif()
