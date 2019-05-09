config_options.mak:
							cp config_options.sample config_options.mak

include						config_options.mak
include						config_common.mak

TARGET						= ca

SUBTARGETS					= libGPUTracking
SUBTARGETS_CLEAN			= libGPUTracking.*

ifeq ($(BUILD_CUDA), 1)
SUBTARGETS					+= libGPUTrackingCUDA
endif
SUBTARGETS_CLEAN			+= libGPUTrackingCUDA.*

ifeq ($(BUILD_OPENCL), 1)
SUBTARGETS					+= libGPUTrackingOCL
endif
SUBTARGETS_CLEAN			+= libGPUTrackingOCL.*

ifeq ($(BUILD_HIP), 1)
SUBTARGETS					+= libGPUTrackingHIP
endif
SUBTARGETS_CLEAN			+= libGPUTrackingHIP.*

CXXFILES					+= standalone.cxx

LIBSUSE						+= -lGPUTracking
ca:							libGPUTracking.so

ca subbuild/libGPUTrackingCUDA.mak subbuild/libGPUTrackingOCL.mak subbuild/libGPUTrackingHIP.mak:	subbuild/libGPUTracking.mak
