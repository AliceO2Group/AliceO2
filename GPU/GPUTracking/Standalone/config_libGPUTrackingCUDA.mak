include						config_options.mak
include						config_common.mak

TARGET						= libGPUTrackingCUDA
TARGETTYPE					= LIB

CUFILES						= Base/cuda/GPUReconstructionCUDA.cu

CONFIG_CUDA					= 1

ALLDEP						+= config_common.mak

DEFINES						+= GPUCA_GPULIBRARY=CUDA

ifneq (${CONFIG_O2DIR}, )
INCLUDEPATHS				+= ${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/cuda/src/
endif

LIBSUSE						+= -lGPUTracking
libGPUTrackingCUDA.so:	libGPUTracking.so
