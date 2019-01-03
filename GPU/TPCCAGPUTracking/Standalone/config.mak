config_options.mak:
							cp config_options.sample config_options.mak

include						config_options.mak
include						config_common.mak

TARGET						= ca

ifeq ($(BUILD_CUDA), 1)
SUBTARGETS					+= libTPCCAGPUTrackingCUDA
DEFINES						+= BUILD_CUDA
endif
SUBTARGETS_CLEAN			+= libTPCCAGPUTrackingCUDA.*

ifeq ($(BUILD_OPENCL), 1)
SUBTARGETS					+= libTPCCAGPUTrackingOCL
DEFINES						+= BUILD_OPENCL
endif
SUBTARGETS_CLEAN			+= libTPCCAGPUTrackingOCL.*

CXXFILES					+= standalone.cxx \
								$(HLTCA_STANDALONE_CXXFILES) \
								$(HLTCA_MERGER_CXXFILES) \
								$(HLTCA_TRD_CXXFILES)

CPPFILES					+= cmodules/qconfig.cpp

ifeq ($(BUILD_EVENT_DISPLAY), 1)
CPPFILES					+= display/AliGPUCADisplay.cpp display/AliGPUCADisplayBackend.cpp display/AliGPUCADisplayBackendGlut.cpp display/AliGPUCADisplayBackendNone.cpp display/AliGPUCADisplayInterpolation.cpp display/AliGPUCADisplayQuaternion.cpp
CONFIG_OPENGL				= 1
CONFIG_X11					= 1
DEFINES						+= BUILD_EVENT_DISPLAY
ifeq ($(ARCH_CYGWIN), 1)
CPPFILES					+= display/AliGPUCADisplayBackendWindows.cpp
else
CPPFILES					+= display/AliGPUCADisplayBackendX11.cpp
endif
endif

ifeq ($(BUILD_QA), 1)
CPPFILES					+= qa/qa.cpp qa/genEvents.cpp
DEFINES						+= BUILD_QA
endif

ifeq ($(LINK_ROOT), 1)
INCLUDEPATHSSYSTEM			+= $(shell root-config --incdir)
LIBSUSE						+= $(shell root-config --libs)
endif

ifneq (${CONFIG_O2DIR}, )
CXXFILES					+= ${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/src/PrimaryVertexContext.cxx \
								${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/src/Cluster.cxx \
								${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/src/TrackerTraitsCPU.cxx \
								${CONFIG_O2DIR}/Detectors/TRD/base/src/TRDGeometryBase.cxx
endif
