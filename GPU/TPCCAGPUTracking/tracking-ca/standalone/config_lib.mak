#Architecture Settings
INTELARCH					= SSE4.2
CUDAVERSION					= 13
CUDAREGS					= 64
ARCHBITS					= 64

TARGET						= cagpu
TARGETTYPE					= LIB

CC_x86_64-pc-linux-gnu		= GCC
CC_i686-pc-cygwin			= ICC

CPPFILES					= 
CXXFILES					= code/AliHLTTPCCATracker.cxx \
							  code/AliHLTTPCCASliceData.cxx \
							  code/AliHLTTPCCASliceOutput.cxx \
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
							  standalone/AliHLTLogging.cxx
							  
CUFILES						= code/AliHLTTPCCAGPUTrackerNVCC.cu
ASMFILES					= 

INCLUDEPATHS				= include code base
DEFINES						= HLTCA_STANDALONE BUILD_GPU

EXTRAFLAGSGCC				= -Weffc++

INTELFLAGSUSE				= $(INTELFLAGSOPT)
VSNETFLAGSUSE				= $(VSNETFLAGSOPT)
GCCFLAGSUSE					= $(GCCFLAGSOPT)
NVCCFLAGSUSE				= $(NVCCFLAGSOPT)
