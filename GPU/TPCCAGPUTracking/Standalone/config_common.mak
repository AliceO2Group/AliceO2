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

INCLUDEPATHS				= include SliceTracker HLTHeaders Merger GlobalTracker TRDTracking Common TPCFastTransformation
DEFINES						= HLTCA_STANDALONE HLTCA_ENABLE_GPU_TRACKER
CPPFILES					= cmodules/timer.cpp

EXTRAFLAGSGCC				=
EXTRAFLAGSLINK				= -rdynamic

ifeq ($(BUILD_DEBUG), 1)
COMPILER_FLAGS				= DBG
else
COMPILER_FLAGS				= OPT
endif
CONFIG_LTO					= 1

CXXFILES					= SliceTracker/AliHLTTPCCASliceData.cxx \
								SliceTracker/AliHLTTPCCASliceOutput.cxx \
								SliceTracker/AliHLTTPCCATracker.cxx \
								SliceTracker/AliHLTTPCCARow.cxx \
								SliceTracker/AliHLTTPCCANeighboursFinder.cxx \
								SliceTracker/AliHLTTPCCANeighboursCleaner.cxx \
								SliceTracker/AliHLTTPCCAGrid.cxx \
								SliceTracker/AliHLTTPCCATrackletConstructor.cxx \
								SliceTracker/AliHLTTPCCATrackletSelector.cxx \
								SliceTracker/AliHLTTPCCAStartHitsFinder.cxx \
								SliceTracker/AliHLTTPCCAHitArea.cxx \
								SliceTracker/AliHLTTPCCATrackParam.cxx \
								SliceTracker/AliHLTTPCCAClusterData.cxx \
								GlobalTracker/AliGPUReconstruction.cxx \
								GlobalTracker/AliGPUReconstructionDeviceBase.cxx \
								GlobalTracker/AliGPUReconstructionConvert.cxx \
								GlobalTracker/AliGPUCAParam.cxx \
								GlobalTracker/AliGPUCASettings.cxx \
								TPCFastTransformation/TPCFastTransform.cxx \
								TPCFastTransformation/TPCDistortionIRS.cxx \
								TPCFastTransformation/IrregularSpline1D.cxx \
								TPCFastTransformation/IrregularSpline2D3D.cxx

HLTCA_MERGER_CXXFILES		= Merger/AliHLTTPCGMMerger.cxx \
								Merger/AliHLTTPCGMSliceTrack.cxx \
								Merger/AliHLTTPCGMPhysicalTrackModel.cxx \
								Merger/AliHLTTPCGMPolynomialField.cxx \
								Merger/AliHLTTPCGMPolynomialFieldManager.cxx \
								Merger/AliHLTTPCGMPropagator.cxx \
								Merger/AliHLTTPCGMTrackParam.cxx

HLTCA_TRD_CXXFILES			= TRDTracking/AliHLTTRDTrack.cxx \
								TRDTracking/AliHLTTRDTracker.cxx \
								TRDTracking/AliHLTTRDTrackletWord.cxx

HLTCA_STANDALONE_CXXFILES	= SliceTracker/AliHLTTPCCATrack.cxx \
								SliceTracker/AliHLTTPCCATracklet.cxx \
								SliceTracker/AliHLTTPCCAMCPoint.cxx

CONFIG_CPP					= gnu++17
CONFIG_CPP_CUDA				= c++14

ifeq ($(ARCH_CYGWIN), 1)
CONFIG_O2DIR				=
BUILD_QA					= 0
endif

ifeq ($(CONFIG_OPENMP), 1)
DEFINES						+= HLTCA_HAVE_OPENMP
endif

ifeq ($(CONFIG_VC), 1)
LIBSUSE						+= -lVc
else
DEFINES						+= HLTCA_NO_VC
endif

ifeq ($(LINK_ROOT), 0)
ifneq ($(CONFIG_O2DIR), )
$(warning Cannot use O2DIR without QA)
CONFIG_O2DIR =
endif
ifeq ($(BUILD_QA), 1)
$(warning Cannot build QA without ROOT)
BUILD_QA = 0
endif
endif

ifeq ($(CONFIG_O2DIR), )
ifeq ($(CONFIG_O2), 1)
$(warning Cannot build v.s. O2 wihout O2 dir)
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
								${CONFIG_O2DIR}/DataFormats/Detectors/TPC/include
endif

ifeq ($(CONFIG_O2), 1)
DEFINES						+= HLTCA_TPC_GEOMETRY_O2
endif

ALLDEP						+= config_common.mak config_options.mak
