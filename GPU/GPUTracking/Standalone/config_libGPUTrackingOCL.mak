include						config_options.mak
include						config_common.mak

TARGET						= libGPUTrackingOCL
TARGETTYPE					= LIB

CXXFILES					= Base/opencl/GPUReconstructionOCL1.cxx \
							  Base/opencl-common/GPUReconstructionOCL.cxx

CLFILES						= Base/opencl-common/GPUReconstructionOCL.cl

INCLUDEPATHS				+= Base/opencl-common

CONFIG_OPENCL				= 1
OPENCL_OPTIONS				= -x clc++
OPENCL_ENVIRONMENT			= GPUCA_FORCE_64BIT_PTR=1

ALLDEP						+= config_common.mak

DEFINES						+= GPUCA_GPULIBRARY=OCL

LIBSUSE						+= -lGPUTracking
libGPUTrackingOCL.so:		libGPUTracking.so
