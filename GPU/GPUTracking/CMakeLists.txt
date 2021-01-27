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

#set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}} -O0") # to uncomment if needed, tired of typing this...

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  if(ENABLE_CUDA OR ENABLE_OPENCL1 OR ENABLE_OPENCL2 OR ENABLE_HIP)
    cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
    include(FeatureSummary)
    find_package(O2GPU)
  endif()
else()
  if(OPENGL_FOUND
     AND GLFW_FOUND
     AND ((ALIGPU_BUILD_TYPE STREQUAL "O2" AND TARGET AliceO2::DebugGUI) OR GLEW_FOUND)
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
    SliceTracker/GPUTPCRow.cxx
    SliceTracker/GPUTPCGlobalTracking.cxx
    SliceTracker/GPUTPCCreateSliceData.cxx
    Merger/GPUTPCGMMerger.cxx
    Merger/GPUTPCGMSliceTrack.cxx
    Merger/GPUTPCGMTrackParam.cxx
    Merger/GPUTPCGMPropagator.cxx
    Merger/GPUTPCGMPhysicalTrackModel.cxx
    Merger/GPUTPCGMPolynomialField.cxx
    Merger/GPUTPCGMPolynomialFieldManager.cxx
    TRDTracking/GPUTRDTrack.cxx
    TRDTracking/GPUTRDTracker.cxx
    TRDTracking/GPUTRDTrackletWord.cxx
    TRDTracking/GPUTRDTrackerKernels.cxx
    Base/GPUParam.cxx)

set(SRCS_NO_CINT
    DataTypes/GPUDataTypes.cxx
    Base/GPUReconstruction.cxx
    Base/GPUReconstructionCPU.cxx
    Base/GPUProcessor.cxx
    Base/GPUMemoryResource.cxx
    Base/GPUGeneralKernels.cxx
    Base/GPUReconstructionDeviceBase.cxx
    Base/GPUReconstructionConvert.cxx
    Base/GPUKernelDebugOutput.cxx
    Global/GPUChain.cxx
    Global/GPUChainTracking.cxx
    Global/GPUTrackingInputProvider.cxx
    Global/GPUErrors.cxx
    Merger/GPUTPCGMMergerGPU.cxx
    utils/timer.cxx)

set(SRCS_NO_H SliceTracker/GPUTPCTrackerDump.cxx
              Merger/GPUTPCGMMergerDump.cxx
              Global/GPUChainTrackingClusterizer.cxx
              Global/GPUChainTrackingTransformation.cxx
              Global/GPUChainTrackingTRD.cxx
              Global/GPUChainTrackingRefit.cxx
              Global/GPUChainTrackingSliceTracker.cxx
              Global/GPUChainTrackingMerger.cxx
              Global/GPUChainTrackingCompression.cxx
              Global/GPUChainTrackingDebugAndProfiling.cxx
              Global/GPUChainTrackingIO.cxx)

set(HDRS_INSTALL
    SliceTracker/GPUTPCMCInfo.h
    SliceTracker/GPUTPCHit.h
    SliceTracker/GPUTPCHitId.h
    SliceTracker/GPUTPCClusterData.h
    SliceTracker/GPUTPCBaseTrackParam.h
    SliceTracker/GPUTPCTracklet.h
    SliceTracker/GPUTPCSliceOutCluster.h
    SliceTracker/GPUTPCTrackLinearisation.h
    Merger/GPUTPCGMBorderTrack.h
    Merger/GPUTPCGMOfflineStatisticalErrors.h
    Merger/GPUTPCGMMergedTrack.h
    Merger/GPUTPCGMMergedTrackHit.h
    DataTypes/GPUTRDDef.h
    DataTypes/GPUSettings.h
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
    Base/GPUConstantMem.h
    Base/GPUParamRTC.h
    Base/GPUReconstructionKernels.h
    Base/GPUReconstructionIncludesITS.h
    Base/GPUReconstructionHelpers.h
    TPCConvert/GPUTPCConvertImpl.h
    Base/GPUReconstructionKernelMacros.h
    DataTypes/GPUO2FakeClasses.h
    DataTypes/GPUSettingsList.h
    DataTypes/GPUOutputControl.h
    DataTypes/GPUO2DataTypes.h
    DataTypes/GPUHostDataTypes.h
    DataTypes/GPUMemorySizeScalers.h
    DataTypes/GPURawData.h
    DataTypes/GPUdEdxInfo.h
    Base/GPUParam.inc
    Merger/GPUTPCGMMergerTypes.h
    Global/GPUErrorCodes.h
    Global/GPUChainTrackingDefs.h
    qconfigoptions.h
    qa/GPUQAHelper.h
    DataCompression/GPUTPCClusterRejection.h
    Definitions/GPUDef.h
    Definitions/GPUDefConstantsAndSettings.h
    Definitions/GPUDefGPUParameters.h
    Definitions/GPUDefOpenCL12Templates.h
    Definitions/GPULogging.h
    Definitions/GPUDefMacros.h
)

# Sources only for O2
if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  set(SRCS ${SRCS} Interface/GPUO2InterfaceConfigurableParam.cxx)
  set(HDRS_CINT_O2 ${HDRS_CINT_O2} DataTypes/TPCdEdxCalibrationSplines.h Interface/GPUO2InterfaceConfigurableParam.h)
  set(HDRS_CINT_O2_ADDITIONAL DataTypes/GPUSettings.h DataTypes/GPUSettingsList.h DataTypes/GPUDataTypes.h TRDTracking/GPUTRDTrack.h) # Manual dependencies for ROOT dictionary generation
endif()

# Sources for O2 and for Standalone if requested in config file
if(ALIGPU_BUILD_TYPE STREQUAL "O2" OR ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(SRCS_NO_CINT ${SRCS_NO_CINT} display/GPUDisplayBackend.cxx)
endif()
if(ALIGPU_BUILD_TYPE STREQUAL "O2" OR CONFIG_O2_EXTENSIONS)
    set(SRCS_NO_CINT
        ${SRCS_NO_CINT}
        Global/GPUChainITS.cxx
        ITS/GPUITSFitter.cxx
        ITS/GPUITSFitterKernels.cxx
        dEdx/GPUdEdx.cxx
        DataTypes/TPCdEdxCalibrationSplines.cxx
        TPCConvert/GPUTPCConvert.cxx
        TPCConvert/GPUTPCConvertKernel.cxx
        DataCompression/GPUTPCCompression.cxx
        DataCompression/GPUTPCCompressionTrackModel.cxx
        DataCompression/GPUTPCCompressionKernels.cxx
        DataCompression/TPCClusterDecompressor.cxx
        DataCompression/GPUTPCClusterStatistics.cxx
        TPCClusterFinder/GPUTPCClusterFinder.cxx
        TPCClusterFinder/ClusterAccumulator.cxx
        TPCClusterFinder/MCLabelAccumulator.cxx
        TPCClusterFinder/GPUTPCCFCheckPadBaseline.cxx
        TPCClusterFinder/GPUTPCCFStreamCompaction.cxx
        TPCClusterFinder/GPUTPCCFChargeMapFiller.cxx
        TPCClusterFinder/GPUTPCCFPeakFinder.cxx
        TPCClusterFinder/GPUTPCCFNoiseSuppression.cxx
        TPCClusterFinder/GPUTPCCFClusterizer.cxx
        TPCClusterFinder/GPUTPCCFDeconvolution.cxx
        TPCClusterFinder/GPUTPCCFMCLabelFlattener.cxx
        TPCClusterFinder/GPUTPCCFDecodeZS.cxx
        TPCClusterFinder/GPUTPCCFGather.cxx
        TPCClusterFinder/TPCPadGainCalib.cxx
        Refit/GPUTrackingRefit.cxx
        Refit/GPUTrackingRefitKernel.cxx
        Merger/GPUTPCGMO2Output.cxx)

    set(SRCS_NO_H ${SRCS_NO_H}
        TPCClusterFinder/GPUTPCClusterFinderDump.cxx)

    set(HDRS_INSTALL ${HDRS_INSTALL}
                     ITS/GPUITSTrack.h
                     TPCClusterFinder/Array2D.h
                     TPCClusterFinder/CfConsts.h
                     TPCClusterFinder/CfFragment.h
                     TPCClusterFinder/CfUtils.h
                     TPCClusterFinder/ChargePos.h
                     TPCClusterFinder/clusterFinderDefs.h
                     TPCClusterFinder/GPUTPCClusterFinderKernels.h
                     TPCClusterFinder/PackedCharge.h
                     TPCClusterFinder/GPUTPCCFChainContext.h)
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
  set(SRCS_NO_CINT ${SRCS_NO_CINT} utils/qsem.cxx
                   display/GPUDisplay.cxx
                   display/GPUDisplayBackendGlfw.cxx)
  set(SRCS_NO_H ${SRCS_NO_H} display/GPUDisplayQuaternion.cxx
                display/GPUDisplayInterpolation.cxx
                display/GPUDisplayKeys.cxx)
  set(HDRS_INSTALL ${HDRS_INSTALL}
                   display/GPUDisplayExt.h
                   display/GPUDisplayShaders.h)
else()
  set(HDRS_INSTALL ${HDRS_INSTALL} display/GPUDisplay.h)
endif()

if(GPUCA_QA)
  set(SRCS_NO_CINT ${SRCS_NO_CINT} qa/GPUQA.cxx)
else()
  set(HDRS_INSTALL ${HDRS_INSTALL} qa/GPUQA.h)
endif()

# Update HDR variables with files derrived from SRC variables
string(REPLACE ".cxx" ".h" HDRS_CINT_ALIROOT "${SRCS}")
string(REPLACE ".cxx" ".h" HDRS_TMP "${SRCS_NO_CINT}")
set(HDRS_INSTALL ${HDRS_INSTALL} ${HDRS_TMP})
unset(HDRS_TMP)

# the internal copy of ImgGUI has been removed in commit 64a93f3f and there is
# only the optional external package
if (TARGET AliceO2::DebugGUI)
  set(DEBUGGUI_TARGET AliceO2::DebugGUI)
endif()
# Main CMake part for O2
if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 TARGETVARNAME targetName
                 PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                       O2::GPUUtils
                                       O2::DataFormatsTPC
                                       O2::TPCBase
                                       O2::TRDBase
                                       O2::ITStracking
                                       O2::TPCFastTransformation
                                       O2::DetectorsRaw
                                       ${DEBUGGUI_TARGET}
                                       O2::Steer
                 PUBLIC_INCLUDE_DIRECTORIES .
                                            Definitions
                                            DataTypes
                                            Base
                                            SliceTracker
                                            TPCConvert
                                            dEdx
                                            ITS
                                            TRDTracking
                                            qa
                                            display
                                            Global
                                            HLTHeaders
                                            Interface
                                            Merger
                                            Refit
                                            DataCompression
                                            TPCClusterFinder
                 TARGETVARNAME targetName
                 SOURCES ${SRCS} ${SRCS_NO_CINT} ${SRCS_NO_H})

  target_include_directories(
    ${targetName}
    PRIVATE $<TARGET_PROPERTY:O2::Framework,INTERFACE_INCLUDE_DIRECTORIES>)

  o2_target_root_dictionary(${MODULE}
                            HEADERS ${HDRS_CINT_O2} ${HDRS_CINT_O2_ADDITIONAL}
                            LINKDEF GPUTrackingLinkDef_O2.h)

  target_compile_definitions(${targetName} PRIVATE GPUCA_O2_LIB
                             GPUCA_TPC_GEOMETRY_O2 HAVE_O2HEADERS)

  install(FILES ${HDRS_CINT_ALIROOT} ${HDRS_CINT_O2} ${HDRS_INSTALL}
          DESTINATION include/GPU)
  install(DIRECTORY utils
          DESTINATION include/GPU
          FILES_MATCHING PATTERN *.h)

  o2_add_test_root_macro(Standalone/tools/createGeo.C
                         PUBLIC_LINK_LIBRARIES O2::GPUTracking
                         LABELS its COMPILE_ONLY)
  o2_add_test_root_macro(Standalone/tools/createLUT.C
                         PUBLIC_LINK_LIBRARIES O2::GPUTracking
                         LABELS its COMPILE_ONLY)

  add_subdirectory(Interface)
endif()

# Main CMake part for AliRoot
if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  set(HDRS_CINT_ALIROOT ${HDRS_CINT_ALIROOT}
                        Base/GPUO2DataTypes.h)
  add_definitions(-DGPUCA_ALIROOT_LIB)
  include_directories(display qa)
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
                      ${CMAKE_SOURCE_DIR}/GPU/Utils
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Definitions
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/DataTypes
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/SliceTracker
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Merger
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Global
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/TPCConvert
                      ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/dEdx
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
  install(DIRECTORY utils
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
  target_link_libraries(${targetName} PUBLIC ${GLFW_LIBRARIES} OpenGL::GL OpenGL::GLU)
  if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    target_compile_definitions(${targetName} PRIVATE GPUCA_DISPLAY_GL3W GPUCA_DISPLAY_OPENGL_CORE)
  else()
    target_link_libraries(${targetName} PUBLIC ${GLEW_LIBRARIES})
  endif()
endif()

# Add compile definitions and libraries depending on available optional dependencies
if(GPUCA_QA)
  message(STATUS "Building GPU QA")
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_QA)
endif()

if(OpenMP_CXX_FOUND)
  message(STATUS "GPU: Using OpenMP: ${OpenMP_CXX_SPEC_DATE}")
  # Must be private, depending libraries might be compiled by compiler not understanding -fopenmp
  target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
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

if(ALIGPU_BUILD_TYPE STREQUAL "O2" OR ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  if(NOT ${CMAKE_SYSTEM} MATCHES Darwin AND (NOT GPUCA_EVENT_DISPLAY OR (GLUT_FOUND AND GLEW_FOUND AND X11_FOUND)))
    add_subdirectory(Standalone/Benchmark)
  endif()
endif()
