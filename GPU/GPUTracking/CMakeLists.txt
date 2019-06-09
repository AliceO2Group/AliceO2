if(ENABLE_CUDA OR ENABLE_OPENCL OR ENABLE_HIP)
  if(CMAKE_SYSTEM_NAME MATCHES Darwin)
    message(WARNING "GPU Tracking disabled on MacOS")
  else()
    if(ENABLE_CUDA)
      add_subdirectory(Base/cuda)
    endif()
    if(ENABLE_OPENCL)
      add_subdirectory(Base/opencl)
    endif()
    if(ENABLE_HIP)
      add_subdirectory(Base/hip)
    endif()
  endif()
endif()

o2_add_library(GPUTracking
               PUBLIC_LINK_LIBRARIES O2::GPUCommon O2::DataFormatsTPC
                                     O2::TRDBase O2::ITStracking
                                     O2::TPCFastTransformation
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
               SOURCES SliceTracker/GPUTPCTrack.cxx
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
                       Base/GPUParam.cxx
                       Interface/GPUO2Interface.cxx
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
                       DataCompression/GPUTPCClusterStatistics.cxx
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
                       Global/GPUChainTrackingDebugAndProfiling.cxx
                       Standalone/utils/timer.cpp
                       SliceTracker/GPUTPCTrackerDump.cxx)

target_compile_definitions(GPUTracking PRIVATE GPUCA_O2_LIB
                           GPUCA_TPC_GEOMETRY_O2 HAVE_O2HEADERS)

if(GPUCA_EVENT_DISPLAY)
  message(STATUS "Building GPU Event Display")
  target_sources(GPUTracking
                 Standalone/utils/qsem.cpp
                 Standalone/display/GPUDisplay.cpp
                 Standalone/display/GPUDisplayBackendGlfw.cpp
                 Standalone/display/GPUDisplayQuaternion.cpp
                 Standalone/display/GPUDisplayInterpolation.cpp
                 Standalone/display/GPUDisplayKeys.cpp)
  target_compile_definitions(GPUTracking PRIVATE BUILD_EVENT_DISPLAY)
  target_link_libraries(GPUTracking -lGL -lGLU -lGLEW)
endif()

if(GPUCA_QA)
  message(STATUS "Building GPU QA")
  target_sources(GPUTracking Standalone/qa/GPUQA.cpp)
  target_compile_definitions(GPUTracking PRIVATE BUILD_QA)
endif()

if(OpenMP_CXX_FOUND)
  message(STATUS "GPU: Using OpenMP: ${OpenMP_CXX_SPEC_DATE}")
  target_compile_definitions(GPUTracking PRIVATE GPUCA_HAVE_OPENMP)
  target_link_libraries(GPUTracking OpenMP::OpenMP_CXX)
endif()

o2_target_root_dictionary(GPUTracking
                          HEADERS Interface/GPUO2Interface.h
                          LINKDEF GPUTrackingLinkDef_O2.h)

o2_add_test(GPUTracking
            PUBLIC_LINK_LIBRARIES O2::GPUTracking
            SOURCES ctest/testGPUTracking.cxx
            COMPONENT_NAME GPU
            LABELS gpu)
