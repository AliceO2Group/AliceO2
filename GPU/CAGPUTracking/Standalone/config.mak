config_options.mak:
							cp config_options.sample config_options.mak

include						config_options.mak
include						config_common.mak

TARGET						= ca

SUBTARGETS					= libCAGPUTracking
SUBTARGETS_CLEAN			= libCAGPUTracking.*

ifeq ($(BUILD_CUDA), 1)
SUBTARGETS					+= libCAGPUTrackingCUDA
endif
SUBTARGETS_CLEAN			+= libCAGPUTrackingCUDA.*

ifeq ($(BUILD_OPENCL), 1)
SUBTARGETS					+= libCAGPUTrackingOCL
endif
SUBTARGETS_CLEAN			+= libCAGPUTrackingOCL.*

ifeq ($(BUILD_HIP), 1)
SUBTARGETS					+= libCAGPUTrackingHIP
endif
SUBTARGETS_CLEAN			+= libCAGPUTrackingHIP.*

CXXFILES					+= standalone.cxx

LIBSUSE						+= -lCAGPUTracking
ca:							libCAGPUTracking.so
