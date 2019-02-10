include						config_options.mak
include						config_common.mak

TARGET						= libCAGPUTrackingCUDA
TARGETTYPE					= LIB

CUFILES						= GlobalTracker/cuda/AliGPUReconstructionCUDA.cu

CONFIG_CUDA					= 1

ALLDEP						+= config_common.mak

DEFINES						+= GPUCA_GPULIBRARY=CUDA

ifneq (${CONFIG_O2DIR}, )
INCLUDEPATHS				+= ${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/cuda/src/
endif

LIBSUSE						+= -lCAGPUTracking
libCAGPUTrackingCUDA.so:	libCAGPUTracking.so
