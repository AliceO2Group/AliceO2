config_options.mak:
							cp config_options.sample config_options.mak

include						config_options.mak
include						config_common.mak

TARGET						= ca
EXTRAFLAGSLINK					+= -Wl,--no-undefined

ifeq ($(BUILD_CUDA), 1)
SUBTARGETS					+= libCAGPUTrackingCUDA
DEFINES						+= BUILD_CUDA
endif
SUBTARGETS_CLEAN			+= libCAGPUTrackingCUDA.*

ifeq ($(BUILD_OPENCL), 1)
SUBTARGETS					+= libCAGPUTrackingOCL
DEFINES						+= BUILD_OPENCL
endif
SUBTARGETS_CLEAN			+= libCAGPUTrackingOCL.*

ifeq ($(BUILD_HIP), 1)
SUBTARGETS					+= libCAGPUTrackingHIP
DEFINES						+= BUILD_HIP
endif
SUBTARGETS_CLEAN			+= libCAGPUTrackingHIP.*


CXXFILES					+= standalone.cxx \
								GlobalTracker/AliGPUReconstructionTimeframe.cxx \
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
