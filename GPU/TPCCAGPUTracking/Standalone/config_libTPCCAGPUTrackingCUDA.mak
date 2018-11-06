include						config_options.mak
include						config_common.mak

TARGET						= libTPCCAGPUTrackingCUDA
TARGETTYPE					= LIB

CXXFILES					=
CUFILES						= GlobalTracker/cuda/AliHLTTPCCAGPUTrackerNVCC.cu \
								GlobalTracker/cuda/AliGPUReconstructionCUDA.cu
ASMFILES					=

CONFIG_CUDA					= 1

ALLDEP						+= config_common.mak
