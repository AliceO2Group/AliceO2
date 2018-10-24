include						config_options.mak
include						config_common.mak

TARGET						= libAliHLTTPCCAGPUSACUDA
TARGETTYPE					= LIB

CXXFILES					= 
CUFILES						= GlobalTracker/cuda/AliHLTTPCCAGPUTrackerNVCC.cu
ASMFILES					=

CONFIG_CUDA					= 1

ALLDEP						+= config_common.mak
