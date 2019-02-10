include						config_options.mak
include						config_common.mak

TARGET						= libCAGPUTrackingHIP
TARGETTYPE					= LIB

HIPFILES					= GlobalTracker/hip/AliGPUReconstructionHIP.hip.cpp

CONFIG_HIP					= 1

ALLDEP						+= config_common.mak

DEFINES						+= GPUCA_GPULIBRARY=HIP

LIBSUSE						+= -lCAGPUTracking
libCAGPUTrackingHIP.so:		libCAGPUTracking.so
