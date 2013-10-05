include						config_common.mak

TARGET						= libAliHLTTPCCAGPUSAOpenCL
TARGETTYPE					= LIB

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
							  cagpubuild/AliHLTTPCCAGPUTrackerBase.cxx \
                              cagpubuild/AliHLTTPCCAGPUTrackerOpenCL.cxx
ASMFILES					= 
CLFILES						= cagpubuild/AliHLTTPCCAGPUTrackerOpenCL.cl

CONFIG_OPENCL				= 1
OPENCL_OPTIONS				= -x clc++

ALLDEP						+= config_common.mak
