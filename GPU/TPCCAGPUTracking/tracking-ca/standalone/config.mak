#Architecture Settings
INTELARCH					= SSE4.2
CUDAVERSION					= 13
CUDAREGS					= 64

TARGET						= ca

CPPFILES					= 
CXXFILES					= standalone.cxx \
								code/AliHLTTPCCATrack.cxx \
								code/AliHLTTPCCATrackParam.cxx \
								code/AliHLTTPCCATracklet.cxx \
								code/AliHLTTPCCAStartHitsFinder.cxx \
								code/AliHLTTPCCANeighboursCleaner.cxx \
								code/AliHLTTPCCAParam.cxx \
								code/AliHLTTPCCAOutTrack.cxx \
								code/AliHLTTPCCATracker.cxx \
								code/AliHLTTPCCASliceData.cxx \
								code/AliHLTTPCCASliceOutput.cxx \
								code/AliHLTTPCCAStandaloneFramework.cxx \
								code/AliHLTTPCCATrackletConstructor.cxx \
								code/AliHLTTPCCANeighboursFinder.cxx \
								code/AliHLTTPCCAGrid.cxx \
								code/AliHLTTPCCATrackletSelector.cxx \
								code/AliHLTTPCCAHitArea.cxx \
								code/AliHLTTPCCAMCPoint.cxx \
								code/AliHLTTPCCAMerger.cxx \
								code/AliHLTTPCCAClusterData.cxx \
								code/AliHLTTPCCARow.cxx
CUFILES						= code/AliHLTTPCCAGPUTracker.cu
ASMFILES					= 

INCLUDEPATHS				= code base include
DEFINES						= HLTCA_STANDALONE BUILD_GPU

INTELFLAGSUSE				= $(INTELFLAGSDBG)
VSNETFLAGSUSE				= $(VSNETFLAGSOPT)
GCCFLAGSUSE					= $(GCCFLAGSDBG)
NVCCFLAGSUSE				= $(NVCCFLAGSDBG)