include						config_options.mak
include						config_common.mak

TARGET						= libAliHLTTPCCAGPUSAOpenCL
TARGETTYPE					= LIB

CXXFILES					= GlobalTracker/AliHLTTPCCAGPUTrackerBase.cxx \
						   GlobalTracker/opencl/AliHLTTPCCAGPUTrackerOpenCL.cxx

ASMFILES					=
CLFILES						= GlobalTracker/opencl/AliHLTTPCCAGPUTrackerOpenCL.cl

CONFIG_OPENCL				= 1
OPENCL_OPTIONS				= -x clc++
OPENCL_ENVIRONMENT			= GPU_FORCE_64BIT_PTR=1

ALLDEP						+= config_common.mak
