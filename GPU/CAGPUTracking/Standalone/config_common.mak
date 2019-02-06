#Architecture Settings
INTELARCH					= Host
GCCARCH						= native
MSVCFAVOR					= INTEL64
CUDAVERSION					= 61
CUDAREGS					= 64
ARCHBITS					= 64

HIDEECHO					= @

CC_x86_64-pc-linux-gnu		= GCC
CC_i686-pc-cygwin			= ICC

INCLUDEPATHS				= . SliceTracker HLTHeaders Merger GlobalTracker TRDTracking Common TPCFastTransformation display qa
DEFINES						= GPUCA_STANDALONE GPUCA_ENABLE_GPU_TRACKER

EXTRAFLAGSGCC				+=
EXTRAFLAGSLINK				+= -rdynamic

ifeq ($(BUILD_DEBUG), 1)
COMPILER_FLAGS				= DBG
else
COMPILER_FLAGS				= OPT
endif
CONFIG_LTO					= 1

GPUCA_TRACKER_CXXFILES			= SliceTracker/AliGPUTPCSliceData.cxx \
								SliceTracker/AliGPUTPCSliceOutput.cxx \
								SliceTracker/AliGPUTPCTracker.cxx \
								SliceTracker/AliGPUTPCTrackerDump.cxx \
								SliceTracker/AliGPUTPCRow.cxx \
								SliceTracker/AliGPUTPCNeighboursFinder.cxx \
								SliceTracker/AliGPUTPCNeighboursCleaner.cxx \
								SliceTracker/AliGPUTPCGrid.cxx \
								SliceTracker/AliGPUTPCTrackletConstructor.cxx \
								SliceTracker/AliGPUTPCTrackletSelector.cxx \
								SliceTracker/AliGPUTPCStartHitsFinder.cxx \
								SliceTracker/AliGPUTPCStartHitsSorter.cxx \
								SliceTracker/AliGPUTPCHitArea.cxx \
								SliceTracker/AliGPUTPCTrackParam.cxx \
								SliceTracker/AliGPUTPCClusterData.cxx \
								GlobalTracker/AliGPUReconstruction.cxx \
								GlobalTracker/AliGPUReconstructionImpl.cxx \
								GlobalTracker/AliGPUReconstructionDeviceBase.cxx \
								GlobalTracker/AliGPUReconstructionConvert.cxx \
								GlobalTracker/AliGPUCAParam.cxx \
								GlobalTracker/AliGPUProcessor.cxx \
								GlobalTracker/AliGPUMemoryResource.cxx \
								GlobalTracker/AliGPUCASettings.cxx \
								GlobalTracker/AliGPUGeneralKernels.cxx \
								TPCFastTransformation/TPCFastTransform.cxx \
								TPCFastTransformation/TPCDistortionIRS.cxx \
								TPCFastTransformation/IrregularSpline1D.cxx \
								TPCFastTransformation/IrregularSpline2D3D.cxx

GPUCA_MERGER_CXXFILES		= Merger/AliGPUTPCGMMerger.cxx \
								Merger/AliGPUTPCGMSliceTrack.cxx \
								Merger/AliGPUTPCGMPhysicalTrackModel.cxx \
								Merger/AliGPUTPCGMPolynomialField.cxx \
								Merger/AliGPUTPCGMPolynomialFieldManager.cxx \
								Merger/AliGPUTPCGMPropagator.cxx \
								Merger/AliGPUTPCGMTrackParam.cxx

GPUCA_TRD_CXXFILES			= TRDTracking/AliGPUTRDTrack.cxx \
								TRDTracking/AliGPUTRDTracker.cxx \
								TRDTracking/AliGPUTRDTrackletWord.cxx

GPUCA_STANDALONE_CXXFILES	= SliceTracker/AliGPUTPCTrack.cxx \
								SliceTracker/AliGPUTPCTracklet.cxx \
								SliceTracker/AliGPUTPCMCPoint.cxx

CONFIG_CPP					= c++17
CONFIG_CPP_CUDA				= c++14

ifeq ($(ARCH_CYGWIN), 1)
CONFIG_O2DIR				=
BUILD_QA					= 0
endif

ifeq ($(CONFIG_OPENMP), 1)
DEFINES						+= GPUCA_HAVE_OPENMP
endif

ifeq ($(CONFIG_VC), 1)
LIBSUSE						+= -lVc
else
DEFINES						+= GPUCA_NO_VC
endif

ifeq ($(LINK_ROOT), 0)
ifneq ($(CONFIG_O2DIR), )
$(warning Cannot use O2DIR without ROOT)
CONFIG_O2DIR =
endif
ifeq ($(BUILD_QA), 1)
$(warning Cannot build QA without ROOT)
BUILD_QA = 0
endif
endif

ifeq ($(CONFIG_O2DIR), )
ifeq ($(CONFIG_O2), 1)
$(warning Cannot build for O2 geometry wihout O2 dir)
CONFIG_O2 = 0
endif
endif

ifneq (${CONFIG_O2DIR}, )
DEFINES						+= HAVE_O2HEADERS
INCLUDEPATHS					+= O2Headers \
								${CONFIG_O2DIR}/Detectors/TPC/base/include \
								${CONFIG_O2DIR}/DataFormats/Detectors/TPC/include \
								${CONFIG_O2DIR}/Detectors/TRD/base/include \
								${CONFIG_O2DIR}/Detectors/TRD/base/src \
								${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/include \
								${CONFIG_O2DIR}/Detectors/ITSMFT/ITS/tracking/cuda/include \
								${CONFIG_O2DIR}/Common/Constants/include \
								${CONFIG_O2DIR}/DataFormats/Reconstruction/include \
								${CONFIG_O2DIR}/Common/MathUtils/include \
								${CONFIG_O2DIR}/DataFormats/Detectors/Common/include \
								${CONFIG_O2DIR}/DataFormats/Detectors/TPC/include \
								${CONFIG_O2DIR}/DataFormats/simulation/include
endif

ifeq ($(CONFIG_O2), 1)
DEFINES						+= GPUCA_TPC_GEOMETRY_O2
endif

ALLDEP						+= config_common.mak config_options.mak
