#ifdef __CINT__
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id: STEERBaseLinkDef.h 65235 2013-12-02 15:40:49Z jgrosseo $ */

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ enum   AliLog::EType_t;

#pragma link C++ class AliVParticle+;
#pragma link C++ class AliVTPCseed+;
#pragma link C++ class AliVTrack+;
#pragma link C++ class AliVCluster+;
#pragma link C++ class AliVCaloCells+;
#pragma link C++ class AliVVertex+;
#pragma link C++ class AliVEvent+;
#pragma link C++ class AliVfriendEvent+;
#pragma link C++ class AliVfriendTrack+;
#pragma link C++ class AliVHeader+;
#pragma link C++ class AliVAODHeader+;
#pragma link C++ class AliVEventHandler+;
#pragma link C++ class AliVEventPool+;
#pragma link C++ class AliVCuts+;
#pragma link C++ class AliVVZERO+;
#pragma link C++ class AliVVZEROfriend+;
#pragma link C++ class AliVZDC+;
#pragma link C++ class AliVAD+;
#pragma link C++ class AliCentrality+;
#pragma link C++ class AliEventplane+;

#pragma link C++ class AliMixedEvent+;

#pragma link C++ class AliPID+;
#pragma link C++ class AliLog+;

#pragma link C++ class AliRunTag+;
#pragma link C++ class AliLHCTag+;
#pragma link C++ class AliDetectorTag+;
#pragma link C++ class AliEventTag+;
#pragma link C++ class AliFileTag+;

#pragma link C++ class AliRunTagCuts+;
#pragma link C++ class AliLHCTagCuts+;
#pragma link C++ class AliDetectorTagCuts+;
#pragma link C++ class AliEventTagCuts+;

#pragma link C++ class AliTagCreator+;

#pragma link C++ class AliHeader+;
#pragma link C++ class AliGenEventHeader+;
#pragma link C++ class AliDetectorEventHeader+;
#pragma link C++ class AliGenCocktailEventHeader+;
#pragma link C++ class AliGenPythiaEventHeader+;
#pragma link C++ class AliGenHijingEventHeader+;
#pragma link C++ class AliCollisionGeometry+;
#pragma link C++ class AliGenDPMjetEventHeader+;
#pragma link C++ class AliGenHerwigEventHeader+;
#pragma link C++ class AliGenGeVSimEventHeader+;
#pragma link C++ class AliGenEpos3EventHeader+;
#pragma link C++ class AliGenEposEventHeader+;
#pragma link C++ class AliGenToyEventHeader+;
#pragma link C++ class AliStack+;
#pragma link C++ class AliMCEventHandler+;
#pragma link C++ class AliInputEventHandler+;
#pragma link C++ class AliDummyHandler+;

#pragma link C++ class AliTrackReference+;
#pragma link C++ class AliSysInfo+;

#pragma link C++ class AliMCEvent+;
#pragma link C++ class AliMCParticle+;
#pragma link C++ class AliMCVertex+;

#pragma link C++ class  AliMagF+;
#pragma link C++ class  AliMagWrapCheb+;
#pragma link C++ class  AliCheb3DCalc+;
#pragma link C++ class  AliCheb3D+;
#pragma link C++ class  AliCheb2DStack+;
#pragma link C++ class  AliCheb2DStackF+;
#pragma link C++ class  AliCheb2DStackS+;

#pragma link C++ class  AliMagFast+;

#pragma link C++ class  AliNeutralTrackParam+;

#pragma link C++ class AliCodeTimer+;
#pragma link C++ class AliCodeTimer::AliPair+;

#pragma link C++ class  AliPDG+;

#pragma link C++ class AliTimeStamp+;
#pragma link C++ class AliTriggerScalers+;
#pragma link C++ class AliTriggerScalersRecord+;

#pragma link C++ class  AliExternalTrackParam+;
#pragma link C++ class AliQA+;

#pragma link C++ class AliTRDPIDReference+;
#pragma link C++ class AliTRDPIDParams+;
/* #if ROOT_VERSION_CODE < 0x56300 // ROOT_VERSION(5,99,0) */
// AliTRDPIDThresholds and Centrality are private
#pragma link C++ class AliTRDPIDParams::AliTRDPIDThresholds+;
#pragma link C++ class AliTRDPIDParams::AliTRDPIDCentrality+;
/* #endif */
#pragma link C++ class AliTRDdEdxParams+;
#pragma link C++ class AliTRDPIDResponseObject+;
#pragma link C++ class AliTRDNDFast+;
#pragma link C++ class AliTRDTKDInterpolator+;
#pragma link C++ class AliTRDTKDInterpolator::AliTRDTKDNodeInfo+;
#pragma link C++ class AliITSPidParams+;
#pragma link C++ class AliPIDResponse+;
#pragma link C++ class AliITSPIDResponse+;
#pragma link C++ class AliTPCPIDResponse+;
#pragma link C++ class AliTPCdEdxInfo+;
#pragma link C++ class AliTOFPIDResponse+;
#pragma link C++ class AliTRDPIDResponse+;
#pragma link C++ class AliEMCALPIDResponse+;
#pragma link C++ class AliHMPIDPIDResponse+;
#pragma link C++ class AliHMPIDPIDParams+;
#pragma link C++ class AliPIDCombined+;
#pragma link C++ class AliPIDValues+;
#pragma link C++ class AliDetectorPID+;
#pragma link C++ class AliTOFHeader+;
#pragma link C++ class AliTOFTriggerMask+;

#pragma link C++ class AliDAQ+;
#pragma link C++ class AliRefArray+;

#pragma link C++ class AliOADBObjCache+;
#pragma link C++ class AliOADBContainer+;


#pragma link C++ class AliVMFT+;
#pragma link C++ class AliCounterCollection+;

#pragma link C++ class AliVCaloTrigger+;

#pragma link C++ class AliTOFPIDParams+;
#pragma link C++ class AliProdInfo+;

#pragma link C++ class AliVTrdTrack+;
#pragma link C++ class AliVTrdTracklet+;
#pragma link C++ class AliGenEventHeaderTunedPbPb+;

#pragma link C++ class  AliVTOFHit+;
#pragma link C++ class  AliVTOFMatch+;
#pragma link C++ class AliVTOFcluster+;
#pragma link C++ class AliVMultiplicity+;
#pragma link C++ class AliGenHepMCEventHeader+;
#pragma link C++ class AliMergeableCollection+;
#pragma link C++ class AliMergeableCollectionIterator+;
#pragma link C++ class AliMergeableCollectionProxy+;
#pragma link C++ class AliMergeable+;

#pragma link C++ class AliMultSelectionBase+;

#pragma link C++ class AliVersion+;

#pragma link C++ class AliDataFile+;

#endif
