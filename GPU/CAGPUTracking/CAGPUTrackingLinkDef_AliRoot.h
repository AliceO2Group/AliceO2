#if defined(__CINT__) || defined(__CLING__)

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class AliGPUTPCTrack+;
#pragma link C++ class AliGPUTPCTracklet+;
#pragma link C++ class AliGPUTPCTracker+;
#pragma link C++ class AliGPUTPCBaseTrackParam+;
#pragma link C++ class AliGPUTPCTrackParam+;
#pragma link C++ class AliGPUTPCRow+;
#pragma link C++ class AliGPUTPCGrid+;
#pragma link C++ class AliGPUTPCHitArea+;
#pragma link C++ class AliGPUTPCTrackerComponent+;
#pragma link C++ class AliGPUTPCNeighboursFinder+;
#pragma link C++ class AliGPUTPCNeighboursCleaner+;
#pragma link C++ class AliGPUTPCStartHitsFinder+;
#pragma link C++ class AliGPUTPCTrackletConstructor+;
#pragma link C++ class AliGPUTPCTrackletSelector+;
#pragma link C++ class AliGPUTPCGlobalMergerComponent+;
#pragma link C++ class AliGPUTPCClusterData+;
#pragma link C++ class AliGPUTPCSliceData+;
#pragma link C++ class AliGPUTPCSliceOutput+;
#pragma link C++ class AliGPUTPCGMTrackParam+;
#pragma link C++ class AliGPUTPCGMSliceTrack+;
#pragma link C++ class AliGPUTPCGMMerger+;
#pragma link C++ class AliGPUTPCGMPolynomialField+;
#pragma link C++ class AliGPUTPCGMPropagator+;
#pragma link C++ class AliGPUTPCGMPhysicalTrackModel+;
#pragma link C++ class AliGPUTPCGMPolynomialFieldManager+;
#pragma link C++ class AliHLTTPCClusterStatComponent+;

//#pragma link C++ class AliGPUTRDTrack+; //Templated, should add linkdef for specialization, but with an ifdef for ROOT >= 6 only
//#pragma link C++ class AliGPUTRDTracker+;
#pragma link C++ class AliGPUTRDTrackerComponent+;
//#pragma link C++ class AliGPUTRDTrackletWord+;
#pragma link C++ class AliGPUTRDTrackletReaderComponent+;

#endif
