include						config_options.mak
include						config_common.mak

TARGET						= libTPCCAGPUTrackingOCL
TARGETTYPE					= LIB

CXXFILES					= GlobalTracker/opencl/AliGPUReconstructionOCL.cxx
ASMFILES					=
CLFILES						= GlobalTracker/opencl/AliGPUReconstructionOCL.cl

CONFIG_OPENCL				= 1
OPENCL_OPTIONS				= -x clc++
OPENCL_ENVIRONMENT			= GPU_FORCE_64BIT_PTR=1

ALLDEP						+= config_common.mak
