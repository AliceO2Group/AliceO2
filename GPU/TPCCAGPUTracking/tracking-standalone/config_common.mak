#Architecture Settings
INTELARCH					= Host
GCCARCH						= native
MSVCFAVOR					= INTEL64
CUDAVERSION					= 61
CUDAREGS					= 64
ARCHBITS					= 64

HIDEECHO					= @

CONFIG_OPENMP				= 1

CC_x86_64-pc-linux-gnu		= GCC
CC_i686-pc-cygwin			= ICC

INCLUDEPATHS				= include code base merger-ca cagpubuild
DEFINES						= HLTCA_STANDALONE
CPPFILES				= cmodules/timer.cpp

EXTRAFLAGSGCC				= -Weffc++ -Wno-unused-local-typedefs
EXTRAFLAGSLINK				= -rdynamic

COMPILER_FLAGS				= OPT
CONFIG_LTO					= 1

CXXFILES				= code/AliHLTTPCCASliceData.cxx \
                                          code/AliHLTTPCCASliceOutput.cxx \
                                          code/AliHLTTPCCATracker.cxx \
                                          code/AliHLTTPCCARow.cxx \
                                          code/AliHLTTPCCANeighboursFinder.cxx \
                                          code/AliHLTTPCCANeighboursCleaner.cxx \
                                          code/AliHLTTPCCAGrid.cxx \
                                          code/AliHLTTPCCAParam.cxx \
                                          code/AliHLTTPCCATrackletConstructor.cxx \
                                          code/AliHLTTPCCATrackletSelector.cxx \
                                          code/AliHLTTPCCAStartHitsFinder.cxx \
                                          code/AliHLTTPCCAHitArea.cxx \
                                          code/AliHLTTPCCAGPUTracker.cxx \
                                          code/AliHLTTPCCATrackParam.cxx \
                                          code/AliHLTTPCCAClusterData.cxx \
                                          code/AliHLTTPCCATrackerFramework.cxx \
                                          interface/AliHLTLogging.cxx

HLTCA_MERGER_CXXFILES			= merger-ca/AliHLTTPCGMMerger.cxx \
                                          merger-ca/AliHLTTPCGMSliceTrack.cxx \
                                          merger-ca/AliHLTTPCGMPhysicalTrackModel.cxx \
                                          merger-ca/AliHLTTPCGMPolynomialField.cxx \
                                          merger-ca/AliHLTTPCGMPolynomialFieldCreator.cxx \
                                          merger-ca/AliHLTTPCGMPropagator.cxx \
                                          merger-ca/AliHLTTPCGMTrackParam.cxx


HLTCA_STANDALONE_CXXFILES		= code/AliHLTTPCCATrack.cxx \
                                          code/AliHLTTPCCATracklet.cxx \
                                          code/AliHLTTPCCAStandaloneFramework.cxx \
                                          code/AliHLTTPCCAMCPoint.cxx

CONFIG_CPP				= gnu++14

ifneq (${CONFIG_O2DIR}, )
DEFINES					+= HAVE_O2HEADERS
INCLUDEPATHSSYSTEM			+= ${CONFIG_O2DIR}/Detectors/TPC/base/include ${CONFIG_O2DIR}/DataFormats/TPC/include
endif
