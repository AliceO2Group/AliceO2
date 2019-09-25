include						config_options.mak
include						config_common.mak

TARGET						= libGPUTrackingHIP
TARGETTYPE					= LIB

HIPFILES					= Base/hip/GPUReconstructionHIP.hip.cxx

CONFIG_HIP					= 1

ALLDEP						+= config_common.mak

DEFINES						+= GPUCA_GPULIBRARY=HIP

LIBSUSE						+= -lGPUTracking
libGPUTrackingHIP.so:		libGPUTracking.so

LINK_OVERRIDE				= $(HIPCC) $(LINKFLAGS)
