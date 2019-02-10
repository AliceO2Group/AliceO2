include						config_options.mak
include						config_common.mak

TARGET						= libCAGPUTracking
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
								GlobalTracker/AliGPUReconstruction.cxx \
								GlobalTracker/AliGPUReconstructionImpl.cxx \
								GlobalTracker/AliGPUReconstructionDeviceBase.cxx \
								GlobalTracker/AliGPUReconstructionConvert.cxx \
								GlobalTracker/AliGPUCAParam.cxx \
								GlobalTracker/AliGPUProcessor.cxx \
								GlobalTracker/AliGPUMemoryResource.cxx \
								GlobalTracker/AliGPUCASettings.cxx \
								GlobalTracker/AliGPUGeneralKernels.cxx \
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

CXXFILES					+= 	GlobalTracker/AliGPUReconstructionTimeframe.cxx \
								$(GPUCA_TRACKER_CXXFILES) \
								$(GPUCA_STANDALONE_CXXFILES) \
								$(GPUCA_MERGER_CXXFILES) \
								$(GPUCA_TRD_CXXFILES)

CPPFILES					+= 	cmodules/timer.cpp \
								cmodules/qsem.cpp \
								cmodules/qconfig.cpp

ifeq ($(BUILD_EVENT_DISPLAY), 1)
CPPFILES					+= display/AliGPUCADisplay.cpp display/AliGPUCADisplayBackend.cpp display/AliGPUCADisplayBackendGlut.cpp display/AliGPUCADisplayBackendNone.cpp display/AliGPUCADisplayInterpolation.cpp display/AliGPUCADisplayQuaternion.cpp display/AliGPUCADisplayKeys.cpp
CONFIG_OPENGL				= 1
CONFIG_GLUT					= 1
CONFIG_GLFW					= 1
CONFIG_X11					= 1
DEFINES						+= BUILD_EVENT_DISPLAY
ifeq ($(ARCH_CYGWIN), 1)
CPPFILES					+= display/AliGPUCADisplayBackendWindows.cpp
else
CPPFILES					+= display/AliGPUCADisplayBackendX11.cpp
CPPFILES					+= display/AliGPUCADisplayBackendGlfw.cpp
endif
endif

ifeq ($(BUILD_QA), 1)
CPPFILES					+= qa/AliGPUCAQA.cpp qa/genEvents.cpp
DEFINES						+= BUILD_QA
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
