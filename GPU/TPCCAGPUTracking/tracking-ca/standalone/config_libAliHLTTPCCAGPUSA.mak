include						config_common.mak

TARGET						= libAliHLTTPCCAGPUSA
TARGETTYPE					= LIB
WORKPATHSUFFIX				= $(TARGETTYPE)

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
							  standalone/AliHLTLogging.cxx \
							  standalone/AliHLTTPCTransform.cxx \
							  cagpubuild/AliHLTTPCCAGPUTrackerBase.cxx
							  
CUFILES						= cagpubuild/AliHLTTPCCAGPUTrackerNVCC.cu
ASMFILES					= 

ALLDEP						+= config_common.mak
