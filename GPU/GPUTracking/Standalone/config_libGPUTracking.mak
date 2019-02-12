include						config_options.mak
include						config_common.mak

TARGET						= libGPUTracking
TARGETTYPE					= LIB

ALLDEP						+= config_common.mak

GPUCA_TRACKER_CXXFILES			= SliceTracker/AliGPUTPCSliceData.cxx \
								SliceTracker/AliGPUTPCSliceOutput.cxx \
								SliceTracker/AliGPUTPCTracker.cxx \
								SliceTracker/AliGPUTPCTrackerDump.cxx \
								SliceTracker/AliGPUTPCRow.cxx \
								SliceTracker/AliGPUTPCNeighboursFinder.cxx \
								SliceTracker/AliGPUTPCNeighboursCleaner.cxx \
								SliceTracker/AliGPUTPCGrid.cxx \
								SliceTracker/AliGPUTPCTrackletConstructor.cxx \
								SliceTracker/AliGPUTPCTrackletSelector.cxx \
								SliceTracker/AliGPUTPCStartHitsFinder.cxx \
								SliceTracker/AliGPUTPCStartHitsSorter.cxx \
								SliceTracker/AliGPUTPCHitArea.cxx \
								SliceTracker/AliGPUTPCTrackParam.cxx \
								SliceTracker/AliGPUTPCClusterData.cxx \
								Base/AliGPUReconstruction.cxx \
								Base/AliGPUReconstructionCPU.cxx \
								Base/AliGPUReconstructionDeviceBase.cxx \
								Base/AliGPUReconstructionConvert.cxx \
								Base/AliGPUParam.cxx \
								Base/AliGPUProcessor.cxx \
								Base/AliGPUMemoryResource.cxx \
								Base/AliGPUSettings.cxx \
								Base/AliGPUGeneralKernels.cxx \
								TPCFastTransformation/TPCFastTransform.cxx \
								TPCFastTransformation/TPCDistortionIRS.cxx \
								TPCFastTransformation/IrregularSpline1D.cxx \
								TPCFastTransformation/IrregularSpline2D3D.cxx

GPUCA_MERGER_CXXFILES		= Merger/AliGPUTPCGMMerger.cxx \
								Merger/AliGPUTPCGMSliceTrack.cxx \
								Merger/AliGPUTPCGMPhysicalTrackModel.cxx \
								Merger/AliGPUTPCGMPolynomialField.cxx \
								Merger/AliGPUTPCGMPolynomialFieldManager.cxx \
								Merger/AliGPUTPCGMPropagator.cxx \
								Merger/AliGPUTPCGMTrackParam.cxx \
								Merger/AliGPUTPCGMMergerGPU.cxx

GPUCA_TRD_CXXFILES			= TRDTracking/AliGPUTRDTrack.cxx \
								TRDTracking/AliGPUTRDTracker.cxx \
								TRDTracking/AliGPUTRDTrackletWord.cxx \
								TRDTracking/AliGPUTRDTrackerGPU.cxx

GPUCA_STANDALONE_CXXFILES	= SliceTracker/AliGPUTPCTrack.cxx \
								SliceTracker/AliGPUTPCTracklet.cxx \
								SliceTracker/AliGPUTPCMCPoint.cxx

CXXFILES					+= 	Base/AliGPUReconstructionTimeframe.cxx \
								$(GPUCA_TRACKER_CXXFILES) \
								$(GPUCA_STANDALONE_CXXFILES) \
								$(GPUCA_MERGER_CXXFILES) \
								$(GPUCA_TRD_CXXFILES)

CPPFILES					+= 	utils/timer.cpp \
								utils/qsem.cpp \
								utils/qconfig.cpp

ifeq ($(BUILD_EVENT_DISPLAY), 1)
CPPFILES					+= display/AliGPUDisplay.cpp display/AliGPUDisplayBackend.cpp display/AliGPUDisplayBackendGlut.cpp display/AliGPUDisplayBackendNone.cpp display/AliGPUDisplayInterpolation.cpp display/AliGPUDisplayQuaternion.cpp display/AliGPUDisplayKeys.cpp
CONFIG_OPENGL				= 1
CONFIG_GLUT					= 1
CONFIG_GLFW					= 1
CONFIG_X11					= 1
ifeq ($(ARCH_CYGWIN), 1)
CPPFILES					+= display/AliGPUDisplayBackendWindows.cpp
else
CPPFILES					+= display/AliGPUDisplayBackendX11.cpp
CPPFILES					+= display/AliGPUDisplayBackendGlfw.cpp
endif
endif

ifeq ($(BUILD_QA), 1)
CPPFILES					+= qa/AliGPUQA.cpp qa/genEvents.cpp
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
								${CONFIG_O2DIR}/Detectors/TRD/base/src/TRDGeometryBase.cxx
endif
