include						config_common.mak

TARGET						= libAliHLTTPCCAGPUSAOpenCL
TARGETTYPE					= LIB

CXXFILES					+= cagpubuild/AliHLTTPCCAGPUTrackerBase.cxx \
						   cagpubuild/AliHLTTPCCAGPUTrackerOpenCL.cxx

ASMFILES					= 
CLFILES						= cagpubuild/AliHLTTPCCAGPUTrackerOpenCL.cl

CONFIG_OPENCL				= 1
OPENCL_OPTIONS				= -x clc++
OPENCL_ENVIRONMENT			= GPU_FORCE_64BIT_PTR=1

ALLDEP						+= config_common.mak
