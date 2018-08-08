include						config_common.mak

TARGET						= libAliHLTTPCCAGPUSA
TARGETTYPE					= LIB

ifeq ($(ARCH_CYGWIN), 1)
CXXFILES					+=
endif
CXXFILES					+= GlobalTracker/AliHLTTPCCAGPUTrackerBase.cxx

CUFILES						= GlobalTracker/cuda/AliHLTTPCCAGPUTrackerNVCC.cu
ASMFILES					= 

CONFIG_CUDA					= 1

ALLDEP						+= config_common.mak
