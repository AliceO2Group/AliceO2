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

INCLUDEPATHS				= include SliceTracker HLTHeaders Merger GlobalTracker TRDTracking Common
DEFINES						= HLTCA_STANDALONE HLTCA_ENABLE_GPU_TRACKER
CPPFILES					= cmodules/timer.cpp

EXTRAFLAGSGCC				= -Weffc++
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
								SliceTracker/AliHLTTPCCAParam.cxx \
								SliceTracker/AliHLTTPCCATrackletConstructor.cxx \
								SliceTracker/AliHLTTPCCATrackletSelector.cxx \
								SliceTracker/AliHLTTPCCAStartHitsFinder.cxx \
								SliceTracker/AliHLTTPCCAHitArea.cxx \
								SliceTracker/AliHLTTPCCAGPUTracker.cxx \
								SliceTracker/AliHLTTPCCATrackParam.cxx \
								SliceTracker/AliHLTTPCCAClusterData.cxx \
								SliceTracker/AliHLTTPCCATrackerFramework.cxx

HLTCA_MERGER_CXXFILES		= Merger/AliHLTTPCGMMerger.cxx \
								Merger/AliHLTTPCGMSliceTrack.cxx \
								Merger/AliHLTTPCGMPhysicalTrackModel.cxx \
								Merger/AliHLTTPCGMPolynomialField.cxx \
								Merger/AliHLTTPCGMPolynomialFieldCreator.cxx \
								Merger/AliHLTTPCGMPropagator.cxx \
								Merger/AliHLTTPCGMTrackParam.cxx

HLTCA_TRD_CXXFILES			= TRDTracking/AliHLTTRDTrack.cxx \
								TRDTracking/AliHLTTRDTracker.cxx \
								TRDTracking/AliHLTTRDTrackletWord.cxx

HLTCA_STANDALONE_CXXFILES	= SliceTracker/AliHLTTPCCATrack.cxx \
								SliceTracker/AliHLTTPCCATracklet.cxx \
								SliceTracker/AliHLTTPCCAStandaloneFramework.cxx \
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

ifneq (${CONFIG_O2DIR}, )
DEFINES						+= HAVE_O2HEADERS
INCLUDEPATHSSYSTEM			+= ${CONFIG_O2DIR}/Detectors/TPC/base/include ${CONFIG_O2DIR}/DataFormats/Detectors/TPC/include
endif

ifeq ($(CONFIG_O2), 1)
DEFINES						+= HLTCA_TPC_GEOMETRY_O2
endif
