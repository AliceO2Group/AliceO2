include						config_options.mak
include						config_common.mak

TARGET						= ca

ifeq ($(BUILD_CUDA), 1)
SUBTARGETS					+= libAliHLTTPCCAGPUSA
endif
SUBTARGETS_CLEAN				+= libAliHLTTPCCAGPUSA.*

ifeq ($(BUILD_OPENCL), 1)
SUBTARGETS					+= libAliHLTTPCCAGPUSAOpenCL
endif
SUBTARGETS_CLEAN				+= libAliHLTTPCCAGPUSAOpenCL.Q

CXXFILES					= standalone.cxx \
								code/AliHLTTPCCATrack.cxx \
								code/AliHLTTPCCATrackParam.cxx \
								code/AliHLTTPCCATracklet.cxx \
								code/AliHLTTPCCAStartHitsFinder.cxx \
								code/AliHLTTPCCANeighboursCleaner.cxx \
								code/AliHLTTPCCAParam.cxx \
								code/AliHLTTPCCATracker.cxx \
								code/AliHLTTPCCATrackerFramework.cxx \
								code/AliHLTTPCCASliceData.cxx \
								code/AliHLTTPCCASliceOutput.cxx \
								code/AliHLTTPCCAStandaloneFramework.cxx \
								code/AliHLTTPCCATrackletConstructor.cxx \
								code/AliHLTTPCCANeighboursFinder.cxx \
								code/AliHLTTPCCAGrid.cxx \
								code/AliHLTTPCCATrackletSelector.cxx \
								code/AliHLTTPCCAHitArea.cxx \
								code/AliHLTTPCCAMCPoint.cxx \
								code/AliHLTTPCCAClusterData.cxx \
								code/AliHLTTPCCARow.cxx \
								code/AliHLTTPCCAGPUTracker.cxx \
								merger-ca/AliHLTTPCGMMerger.cxx \
								merger-ca/AliHLTTPCGMSliceTrack.cxx \
								merger-ca/AliHLTTPCGMTrackParam.cxx \
								standalone/AliHLTLogging.cxx

ifeq ($(BUILD_EVENT_DISPLAY), 1)
CPPFILES					+= display/opengl.cpp
CONFIG_OPENGL				= 1
CONFIG_X11					= 1
DEFINES						+= BUILD_EVENT_DISPLAY
endif

ifeq ($(BUILD_QA), 1)
CPPFILES					+= qa/qa.cpp
DEFINES						+= BUILD_QA
INCLUDEPATHSSYSTEM				+= $(shell root-config --incdir)
LIBSUSE						+= $(shell root-config --libs)
endif

ALLDEP						+= config_common.mak

o2:
						make CONFIGFILE=config_o2.mak -f makefile
