include						config_common.mak

TARGET						= libAliHLTTPCCAGPUSA
TARGETTYPE					= LIB

CXXFILES					= cagpubuild/AliHLTTPCCAGPUTrackerBase.cxx

CUFILES						= cagpubuild/AliHLTTPCCAGPUTrackerNVCC.cu
ASMFILES					= 

CONFIG_CUDA					= 1

ALLDEP						+= config_common.mak
