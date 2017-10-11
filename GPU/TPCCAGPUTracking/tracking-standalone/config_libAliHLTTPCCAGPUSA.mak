include						config_common.mak

TARGET						= libAliHLTTPCCAGPUSA
TARGETTYPE					= LIB

CXXFILES					+= cagpubuild/AliHLTTPCCAGPUTrackerBase.cxx

CUFILES						= cagpubuild/AliHLTTPCCAGPUTrackerNVCC.cu
ASMFILES					= 

CONFIG_CUDA					= 1

ALLDEP						+= config_common.mak

GCCCUDA						= /usr/x86_64-pc-linux-gnu/gcc-bin/5.4.0/x86_64-pc-linux-gnu-gcc