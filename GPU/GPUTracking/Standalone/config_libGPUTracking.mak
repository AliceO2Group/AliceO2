include						config_options.mak
include						config_common.mak

TARGET						= libGPUTracking
TARGETTYPE					= LIB

ALLDEP						+= config_common.mak

GPUCA_TRACKER_CXXFILES			= SliceTracker/GPUTPCSliceData.cxx \
								SliceTracker/GPUTPCSliceOutput.cxx \
								SliceTracker/GPUTPCTracker.cxx \
								SliceTracker/GPUTPCTrackerDump.cxx \
								SliceTracker/GPUTPCRow.cxx \
								SliceTracker/GPUTPCNeighboursFinder.cxx \
								SliceTracker/GPUTPCNeighboursCleaner.cxx \
								SliceTracker/GPUTPCGrid.cxx \
								SliceTracker/GPUTPCTrackletConstructor.cxx \
								SliceTracker/GPUTPCTrackletSelector.cxx \
								SliceTracker/GPUTPCStartHitsFinder.cxx \
								SliceTracker/GPUTPCStartHitsSorter.cxx \
								SliceTracker/GPUTPCHitArea.cxx \
								SliceTracker/GPUTPCTrackParam.cxx \
								SliceTracker/GPUTPCClusterData.cxx \
								Base/GPUDataTypes.cxx \
								Base/GPUReconstruction.cxx \
								Base/GPUReconstructionCPU.cxx \
								Base/GPUReconstructionDeviceBase.cxx \
								Base/GPUParam.cxx \
								Base/GPUProcessor.cxx \
								Base/GPUMemoryResource.cxx \
								Base/GPUSettings.cxx \
								Base/GPUGeneralKernels.cxx \
								Base/GPUReconstructionTimeframe.cxx \
								Base/GPUReconstructionConvert.cxx \
								TPCConvert/GPUTPCConvert.cxx \
								TPCConvert/GPUTPCConvertKernel.cxx \
								Global/GPUChain.cxx \
								Global/GPUChainTracking.cxx \
								Global/GPUChainTrackingDebugAndProfiling.cxx \
								TPCFastTransformation/TPCFastTransform.cxx \
								TPCFastTransformation/TPCDistortionIRS.cxx \
								TPCFastTransformation/TPCFastTransformGeo.cxx \
								TPCFastTransformation/IrregularSpline1D.cxx \
								TPCFastTransformation/IrregularSpline2D3D.cxx

GPUCA_MERGER_CXXFILES		= Merger/GPUTPCGMMerger.cxx \
								Merger/GPUTPCGMSliceTrack.cxx \
								Merger/GPUTPCGMPhysicalTrackModel.cxx \
								Merger/GPUTPCGMPolynomialField.cxx \
								Merger/GPUTPCGMPolynomialFieldManager.cxx \
								Merger/GPUTPCGMPropagator.cxx \
								Merger/GPUTPCGMTrackParam.cxx \
								Merger/GPUTPCGMMergerGPU.cxx

GPUCA_TRD_CXXFILES			= TRDTracking/GPUTRDTrack.cxx \
								TRDTracking/GPUTRDTracker.cxx \
								TRDTracking/GPUTRDTrackletWord.cxx \
								TRDTracking/GPUTRDTrackerGPU.cxx
								
GPUCA_ITS_CXXFILES			= ITS/GPUITSFitter.cxx \
								ITS/GPUITSFitterKernels.cxx \
								Global/GPUChainITS.cxx
								
GPUCA_STANDALONE_CXXFILES	= SliceTracker/GPUTPCTrack.cxx \
								SliceTracker/GPUTPCTracklet.cxx \
								SliceTracker/GPUTPCMCPoint.cxx
								
GPUCA_COMPRESSION_FILES		= DataCompression/GPUTPCCompression.cxx \
								DataCompression/GPUTPCCompressionTrackModel.cxx \
								DataCompression/GPUTPCCompressionKernels.cxx \
								DataCompression/TPCClusterDecompressor.cxx \
								DataCompression/GPUTPCClusterStatistics.cxx
								
GPUCA_DEDX_CXXFILES			= dEdx/GPUdEdx.cxx

CXXFILES					+= 	$(GPUCA_TRACKER_CXXFILES) \
								$(GPUCA_STANDALONE_CXXFILES) \
								$(GPUCA_MERGER_CXXFILES) \
								$(GPUCA_TRD_CXXFILES)

CPPFILES					+= 	utils/timer.cpp \
								utils/qsem.cpp \
								utils/qconfig.cpp

ifeq ($(BUILD_EVENT_DISPLAY), 1)
CPPFILES					+= display/GPUDisplay.cpp display/GPUDisplayBackend.cpp display/GPUDisplayBackendGlut.cpp display/GPUDisplayBackendNone.cpp display/GPUDisplayInterpolation.cpp display/GPUDisplayQuaternion.cpp display/GPUDisplayKeys.cpp
CONFIG_OPENGL				= 1
CONFIG_GLUT					= 1
CONFIG_GLFW					= 1
CONFIG_X11					= 1
ifeq ($(ARCH_CYGWIN), 1)
CPPFILES					+= display/GPUDisplayBackendWindows.cpp
else
CPPFILES					+= display/GPUDisplayBackendX11.cpp
CPPFILES					+= display/GPUDisplayBackendGlfw.cpp
endif
endif

ifeq ($(BUILD_QA), 1)
CPPFILES					+= qa/GPUQA.cpp qa/genEvents.cpp
endif

ifeq ($(LINK_ROOT), 1)
INCLUDEPATHSSYSTEM			+= $(shell root-config --incdir)
LIBSUSE						+= $(shell root-config --libs)
endif

ifneq (${CONFIG_O2DIR}, )
CXXFILES					+= ${CONFIG_O2DIR}/DataFormats/simulation/src/MCCompLabel.cxx \
								${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/src/PrimaryVertexContext.cxx \
								${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/src/Cluster.cxx \
								${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/src/ClusterLines.cxx \
								${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/src/TrackerTraitsCPU.cxx \
								${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/src/VertexerTraits.cxx \
								${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/src/ROframe.cxx \
								${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/src/Road.cxx \
								${CONFIG_O2DIR}/Detectors/TRD/base/src/TRDGeometryBase.cxx \
								${CONFIG_O2DIR}/Detectors/Base/src/MatLayerCylSet.cxx \
								${CONFIG_O2DIR}/Detectors/Base/src/MatLayerCyl.cxx \
								${CONFIG_O2DIR}/Detectors/Base/src/Ray.cxx \
								$(GPUCA_ITS_CXXFILES) \
								$(GPUCA_DEDX_CXXFILES) \
								$(GPUCA_COMPRESSION_FILES)

endif
